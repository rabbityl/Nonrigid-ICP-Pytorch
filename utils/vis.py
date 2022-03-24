import os
import time
import copy
import open3d as o3d
import numpy as np

from skimage import io


def meshgrid(H, W):
    '''
    @param H:
    @param W:
    @return:
    '''
    u = np.arange(0, W).reshape(1,-1)
    v = np.arange(0, H).reshape(-1,1)
    u = np.repeat( u, H, axis=0 )
    v = np.repeat( v, W, axis=1 )
    return u, v

def construct_frame_trimesh(point_image ,pix_mask, mesh_emax = 0.1):
    '''
    @param point_image:
    @param pix_mask:
    @param mesh_emax:
    @return:
    '''
    """
    A---B
    |   |
    D---C
    Two triangles for each pixel square: "ADB" and "DCB"
    right-hand rule for rendering
    :return: indexes of triangles
    """

    _, H, W = point_image.shape

    XYZ = point_image #np.moveaxis(point_image, 0, -1)

    index_x, index_y = meshgrid(H, W )
    index_pix = (index_x + index_y * W)

    A_ind = index_pix[ 1:-1, 1:-1]
    B_ind = index_pix[ 1:-1, 2:]
    C_ind = index_pix[ 2:, 2:]
    D_ind = index_pix[ 2:, 1:-1]

    A_msk = pix_mask[ 1:-1, 1:-1]
    B_msk = pix_mask[ 1:-1, 2:]
    C_msk = pix_mask[ 2:, 2:]
    D_msk = pix_mask[2:, 1:-1]

    A_p3d = XYZ[:, 1:-1, 1:-1]
    B_p3d = XYZ[:, 1:-1, 2:]
    C_p3d = XYZ[:, 2:, 2:]
    D_p3d = XYZ[:, 2:, 1:-1]

    AB_dist = np.linalg.norm(A_p3d - B_p3d, axis=0)
    BC_dist = np.linalg.norm(C_p3d - B_p3d, axis=0)
    CD_dist = np.linalg.norm(C_p3d - D_p3d, axis=0)
    DA_dist = np.linalg.norm(A_p3d - D_p3d, axis=0)
    DB_dist = np.linalg.norm(B_p3d - D_p3d, axis=0)

    AB_mask = (AB_dist < mesh_emax) * A_msk * B_msk
    BC_mask = (BC_dist < mesh_emax) * B_msk * C_msk
    CD_mask = (CD_dist < mesh_emax) * C_msk * D_msk
    DA_mask = (DA_dist < mesh_emax) * D_msk * A_msk
    DB_mask = (DB_dist < mesh_emax) * D_msk * B_msk

    ADB_ind = np.stack([A_ind, D_ind, B_ind]).reshape(3, -1)
    DCB_ind = np.stack([D_ind, C_ind, B_ind]).reshape(3, -1)

    ADB_msk = (AB_mask * DB_mask * DA_mask).reshape(-1)
    DCB_msk = (CD_mask * DB_mask * BC_mask).reshape(-1)

    triangles = np.concatenate([ADB_ind, DCB_ind], axis=1)
    triangles_msk = np.concatenate([ADB_msk, DCB_msk])

    valid_triangles = triangles[:, triangles_msk]

    XYZ = np.moveaxis(XYZ, 0, -1).reshape(-1, 3)
    return  XYZ,   valid_triangles.T


def node_o3d_spheres (node_array, r=0.1, resolution=10, color = [0,1,0]):
    '''
    @param node_array: [N, 3]
    @param r:
    @param resolution:
    @param color:
    @return:
    '''

    N, _ = node_array.shape

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=resolution)

    vertices  = np.asarray(mesh_sphere.vertices)   # point 3d
    triangles = np.asarray(mesh_sphere.triangles) # index

    num_sphere_vertex, _ = vertices.shape

    vertices = np.expand_dims (vertices, axis=0)
    triangles = np.expand_dims (triangles, axis=0)

    vertices = np.repeat ( vertices , [N], axis=0) # change corr 3D
    triangles = np.repeat ( triangles ,[N], axis=0) # change index

    # reposition vertices
    node_array = np.expand_dims( node_array , axis=1)
    vertices = node_array + vertices
    vertices = vertices.reshape( [-1, 3])

    # change index
    index_offset = np.arange(N).reshape( N, 1, 1) * num_sphere_vertex
    triangles = triangles + index_offset
    triangles = triangles.reshape( [-1, 3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    mesh.paint_uniform_color(color)

    # # o3d.visualization.draw_geometries([ mesh ])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(mesh)
    # vis.get_render_option().load_from_json("./renderoption.json")
    # vis.run()
    # # vis.capture_screen_image("output/ours_silver-20.jpg")
    # vis.destroy_window()
    return mesh



def save_grayscale_image(filename, image_numpy):
    image_to_save = np.copy(image_numpy)
    image_to_save = (image_to_save * 255).astype(np.uint8)

    if len(image_to_save.shape) == 2:
        io.imsave(filename, image_to_save)
    elif len(image_to_save.shape) == 3:
        assert image_to_save.shape[0] == 1 or image_to_save.shape[-1] == 1
        image_to_save = image_to_save[0]
        io.imsave(filename, image_to_save)



def merge_meshes(meshes):
    # Compute total number of vertices and faces.
    num_vertices = 0
    num_triangles = 0
    num_vertex_colors = 0
    for i in range(len(meshes)):
        num_vertices += np.asarray(meshes[i].vertices).shape[0]
        num_triangles += np.asarray(meshes[i].triangles).shape[0]
        num_vertex_colors += np.asarray(meshes[i].vertex_colors).shape[0]

    # Merge vertices and faces.
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    triangles = np.zeros((num_triangles, 3), dtype=np.int32)
    vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

    vertex_offset = 0
    triangle_offset = 0
    vertex_color_offset = 0
    for i in range(len(meshes)):
        current_vertices = np.asarray(meshes[i].vertices)
        current_triangles = np.asarray(meshes[i].triangles)
        current_vertex_colors = np.asarray(meshes[i].vertex_colors)

        vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
        triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
        vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors

        vertex_offset += current_vertices.shape[0]
        triangle_offset += current_triangles.shape[0]
        vertex_color_offset += current_vertex_colors.shape[0]

    # Create a merged mesh object.
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.paint_uniform_color([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh