import bpy
import mathutils
from mathutils import Vector
import math
import numpy as np
import scipy
from scipy import sparse, linalg, cluster
from scipy.sparse import linalg
import random

# Implementation of the paper:
# Rong Liu and Hao Zhang, "Segmentation of 3D meshes through spectral clustering," 
#12th Pacific Conference on Computer Graphics and Applications, 2004. PG 2004. Proceedings.,
# Seoul, South Korea, 2004, pp. 298-305, doi: 10.1109/PCCGA.2004.1348360.


class meshSegmentation():
    def __init__(self, obj, k=2 ,delta=0.0, eta=0.5):
        self.mesh = obj.data
        self.vertex_groups = obj.vertex_groups
        self.k = k #number of clusters
        # Two important parameters used to account geodesic and angular distances
        # between mesh faces
        # delta is a real number in range(0,1) controlling
        # the weight of geodesic distance and angle distance for
        # mesh segmentation. It should be small (the paper suggest to
        # take it between 0.01 - 0.05) to have a visually meaningful
        # mesh segmentation
        self.delta = delta
        # The eta value small value gives more weight to the concavity
        # the paper suggest to keep it between 0.1 - 0.2
        self.eta = eta
        

    def get_face_center(self, face):
        center = Vector((0,0,0))
        for idx in face.vertices:
            center += self.mesh.vertices[idx].co
        return center/len(face.vertices)

    def get_geodesic_distance(self, face1, face2, edge):
        """
        The geodesic distance is the shortest path between two points
        ref: 
            https://www.sciencedirect.com/topics/computer-science/geodesic-distance#:~:text=A%20simple%20measure%20of%20the,shortest%20path%20between%20the%20vertices.
        """
        v0, v1 = edge[0], edge[1]
        edge_center = (self.mesh.vertices[v0].co + self.mesh.vertices[v1].co)/2
        # The shortest path between face1 and face2 is the distance between
        # the two face centers and the center of the edge between them
        d1 = (edge_center - self.get_face_center(face1)).length
        d2 = (edge_center - self.get_face_center(face2)).length
        return d1 + d2

    def get_angular_distance(self, face1, face2):
        '''Returns the angular distanfe between two faces'''
        
        theta = Vector.angle(face1.normal, face2.normal)
        d = 1 - math.cos(theta)

        #TODO scale convex angles
        if face1.normal.dot(face2.normal) < 0:
            d *= self.eta
        
        return d
    
    def create_distance_matrix(self):
        ''' Generates a distance matrix: [D]_ij = dist(face_i, face_j)'''
        
        faces = self.mesh.polygons
        n = len(faces)
        
        adj_faces_dict = {} #dictionary of adjacency faces
        # find adjacent faces by iterating edges
        # store them in a dictionary using the edge as key
        # and the faces indixes in an array
        for index, face in enumerate(faces):
            for edge in face.edge_keys:
                if edge in adj_faces_dict:
                    adj_faces_dict[edge].append(index)
                else:
                    adj_faces_dict[edge] = [index]

        row_indices = []
        col_indices = []
        geo_distances = []  # geodesic distances 
        angle_distances = []  # angular distances
        # calculate geodesic adn angular distances for all adjiacent faces
        for edge, adj_faces in adj_faces_dict.items():
            if len(adj_faces) == 2:
                # Get indexes of adjiacent faces
                i = adj_faces[0]
                j = adj_faces[1]

                geo_d = self.get_geodesic_distance(faces[i], faces[j], edge)
                angle_d = self.get_angular_distance(faces[i], faces[j])
                geo_distances.append(geo_d)
                angle_distances.append(angle_d)
                row_indices.append(i)
                col_indices.append(j)
                # distance matrix is symmetric
                geo_distances.append(geo_d)
                angle_distances.append(angle_d)
                row_indices.append(j)
                col_indices.append(i)

            elif len(adj_faces) > 2:
                print("Edge with more than 2 adjacent faces: " + str(adj_faces) + "!")
        # convert to numpy arrays for speedup
        geo_distances = np.array(geo_distances)
        angle_distances = np.array(angle_distances)
        # TODO explain this
        values = self.delta * geo_distances / np.mean(geo_distances) + \
                 (1.0 - self.delta) * angle_distances / np.mean(angle_distances)

        # create sparse matrix (similar to matlab sparse command)
        D = scipy.sparse.csr_matrix(
            (values, (row_indices, col_indices)), shape=(n, n))
        return D
    
    def create_affinity_matrix(self):
        '''
        An Affinity Matrix, also called a Similarity Matrix, is an essential statistical 
        technique used to organize the mutual similarities between a set of data points.
        The affinity matrix is obtained by [W]_ij = e^(-dist(face_i, face_j)/ (2 sigma^2)
        '''
        
        n = len(self.mesh.polygons) # number of faces/entries in affinity matrix
        # Compute the distance matrix of all the faces i.e. [D]_ij = dist(face_i, face_j)
        D = self.create_distance_matrix()
        # for each non adjacent pair of faces find shortest path of adjacent faces
        D = scipy.sparse.csgraph.dijkstra(D)
        inf_indices = np.where(np.isinf(D))
        D[inf_indices] = 0 #set inf distances to zero for later computation

        # change distance entries to similarities
        sigma = D.sum()/(n**2) #from paper sigma = sum(D) / n^2
        # [W]_ij = e^(-dist(face_i, face_j)/ (2 sigma^2)
        W = np.exp(-D/ (2 * sigma ** 2) )

        W[inf_indices] = 0 #set inf distances to zero for later computation
        np.fill_diagonal(W, 1) # dist(face_i, face_i) = 1

        return W
        
    def loose_cluster_components(self):
        '''Separates the cluster by creating new objects'''
        
        # Original mesh data
        mesh_polygons = np.array(self.mesh.polygons)        
        mesh_vertices = np.array(self.mesh.vertices)
        
        # To create new object we need 3 things:
        # 1) array vertices (x, y, z)
        # 2) array edges pair containing vertices index
        # 3) array faces containing three or more vertices index
        
        for i in range(self.k):
            polygons = mesh_polygons[self.idx == i]

            verts_idx =  []
            for pol in polygons:
                verts_idx += pol.vertices
            verts = [v.co for v in mesh_vertices[verts_idx]]
            print('Vertices:' , verts)
            
            edges = []
            for pol in polygons:
                for ek in pol.edge_keys:
#                    print(ek)
                    edge = self.mesh.edges[self.mesh.edge_keys.index(ek)]
                    edges += [ (edge.vertices[0], edge.vertices[1])  ]
            
            print('Edges:', edges)
            faces = []
            for face in polygons:
                verts_idx = [v for v in face.vertices]                
                faces += [verts_idx]
            print('Faces:', faces)

            new_mesh = bpy.data.meshes.new(name='piece' + str(i))
            new_mesh.from_pydata(verts, edges, faces)
            new_mesh.update()
            
            new_object = bpy.data.objects.new('new_object', new_mesh)
            bpy.context.collection.objects.link(new_object)

           
    def assign_colors(self):
        
        self.mesh.materials.clear()
        
        for i in range(self.k):
            mat_name = 'mat'+self.mesh.name + str(i)
            material = bpy.data.materials.new(mat_name)
            material.diffuse_color = (random.random(), random.random(),
                                      random.random(), 1.0)
            self.mesh.materials.append(material)

        for i, id in enumerate(self.idx):
            self.mesh.polygons[i].material_index = id
    
    def assign_vertex_groups(self):
        '''Assign vertex to different clusters'''
        self.vertex_groups.clear() 
        for i in range(self.k):
            cluster = self.vertex_groups.new( name = 'Cluster_' + str(i) )
            faces = np.array(self.mesh.polygons)
#            print(self.idx)
            cluster_faces = faces[self.idx == i]
            verts = []
            for face in cluster_faces:
                verts += face.vertices
            cluster.add( verts, 0, 'REPLACE' )


    def run(self):
        '''Runs the spectral partitioning on the mesh and returns the partitions'''

        W = self.create_affinity_matrix()
        # normalise the affinity matrix
        # D (not to be confused with the distance matrix!) is a diagonal
        # degree matrix whose i-th diagonal is the sum of W i-th row 
        D = W.sum(1)
        D_sqrt_rec = np.sqrt(np.reciprocal(D))
        N = (D_sqrt_rec.transpose() * W).transpose() * D_sqrt_rec
        # Compute eigenvectors
#        _, V = scipy.linalg.eigh(N, eigvals = (N.shape[0] - self.k, N.shape[0] - 1))
        _, V = scipy.sparse.linalg.eigsh(N, self.k)
        # Normalise eigenvectors
        V /= np.linalg.norm(V, axis=1)[:,None]
        # use k-means
        _, self.idx = scipy.cluster.vq.kmeans2(V, self.k, minit='++', iter=50)
#        print(self.idx)
        print(len(self.idx))
        
        print('Done mesh segmentation!')
        
# Testing
mesh_segmentation = meshSegmentation(bpy.context.active_object, k=2)
mesh_segmentation.run()
mesh_segmentation.assign_vertex_groups()
mesh_segmentation.assign_colors()
# mesh_segmentation.loose_cluster_components() #buggy