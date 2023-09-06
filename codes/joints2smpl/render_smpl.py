import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
import argparse
from tqdm import tqdm
import argparse
import time
import pickle

import numpy as np
import trimesh
import pyrender
from pyrender.constants import RenderFlags
import imageio
import smplx
import h5py
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from smplify import SMPLify3D
import config

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1,
                    help='input batch size')
parser.add_argument('--num_smplify_iters', type=int, default=50,
                    help='num of smplify iters')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--gpu_ids', type=int, default=0,
                    help='choose gpu ids')
parser.add_argument('--num_joints', type=int, default=22,
                    help='joint number')
parser.add_argument('--joint_category', type=str, default="AMASS",
                    help='use correspondence')
parser.add_argument('--file_name', type=str, default="./demo/demo_data/whisper_4.npy",
                    help='data in the folder')
parser.add_argument('--save_dir', type=str, default="./demo/demo_results/",
                    help='results save folder')
parser.add_argument('--width', type=int, default=512,
                    help='width of output image')
parser.add_argument('--height', type=int, default=512,
                    help='height of output image')
opt = parser.parse_args()

def get_smpl_faces():
    return np.load(os.path.join(config.SMPL_MODEL_DIR, "smplfaces.npy"))




class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, background=None, resolution=(224, 224), bg_color=[0, 0, 0, 0.5], orig_img=False, wireframe=False):
        width, height = resolution
        self.background = np.zeros((height, width, 3))
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=0.5
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose.copy())


    def render(self, img, verts, verts2, cam, angle=None, axis=None, mesh_filename=None, color_1=[1.0, 1.0, 0.9],color_2=[1.0, 1.0, 0.9]):
        # load two meshes
        mesh_1 = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        mesh_2 = trimesh.Trimesh(vertices=verts2, faces=self.faces, process=False)


        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=10000000000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=(color_1[0], color_1[1], color_1[2], 1.0)
        )
        material_2 = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=(color_2[0], color_2[1], color_2[2], 1.0)
        )

        # add both meshes to the scene
        mesh = pyrender.Mesh.from_trimesh(mesh_1, material=material)
        mesh_node = self.scene.add(mesh, 'mesh')
        mesh2 = pyrender.Mesh.from_trimesh(mesh_2, material=material_2)
        mesh_node_2 = self.scene.add(mesh2, 'mesh')
        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(mesh_node_2)
        self.scene.remove_node(cam_node)

        return image


def get_renderer(width, height):
    renderer = Renderer(resolution=(width, height),
                        bg_color=[1, 1, 1, 0.5],
                        orig_img=False,
                        wireframe=False)
    return renderer



def render_video(meshes_1,meshes_2, renderer, savepath, background, cam=(1, 1,0.0, 0.5)):
    writer = imageio.get_writer(savepath, duration=1000*1/30)
    
    imgs = []
    color_1 = [0.11, 0.53, 0.8]
    color_2 = [0.8, 0.53, 0.11]
    mesh1 = meshes_1*0.5
    mesh2 = meshes_2*0.5
    for idx in tqdm(range(len(mesh1))):
        # scaled the vertices for better representation
        img = renderer.render(background, mesh1[idx], mesh2[idx], cam, color_1=color_1, color_2=color_2)
        imgs.append(img)

    imgs = np.array(imgs)
    masks = ~(imgs/255. > 0.96).all(-1)

    coords = np.argwhere(masks.sum(axis=0))
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)

    for cimg in imgs[:, y1:y2, x1:x2]:
        writer.append_data(cimg)
    writer.close()



if __name__ == "__main__":

    device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
    #load mean file as initial data
    file = h5py.File(config.SMPL_MEAN_FILE, 'r')
    init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
    init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()

    data_name = os.path.split(os.path.splitext(opt.file_name)[0])[-1] + '.gif'
    dir_save = os.path.join(opt.save_dir, data_name)
    if not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True) 
    out_name = dir_save.replace('.gif','.pkl')
    print(data_name)
    data = np.load(opt.file_name)

    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            out_mesh_1, out_mesh_2 = pickle.load(f)
    else:
        #2 person motion
        num_pers = data.shape[0]
        seq_len = data.shape[1]

        cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)
        pred_pose = torch.zeros(num_pers*data.shape[1], 72).to(device)
        pred_betas = torch.zeros(num_pers*data.shape[1], 10).to(device)
        pred_cam_t = torch.zeros(num_pers*data.shape[1], 3).to(device)
        keypoints_3d = torch.zeros(num_pers*data.shape[1], 22, 3).to(device)
        
        confidence_input =  torch.ones(22)
        # put more confidence into foot and ankle
        confidence_input[7] = 1.5
        confidence_input[8] = 1.5
        confidence_input[10] = 1.5
        confidence_input[11] = 1.5


        smplmodel = smplx.create(config.SMPL_MODEL_DIR, 
                                model_type="smpl", gender="neutral", ext="pkl",
                                batch_size=opt.batchSize).to(device)
        # # #-------------initialize SMPLify
        smplify = SMPLify3D(smplxmodel=smplmodel,
                            batch_size=opt.batchSize,
                            joints_category=opt.joint_category,
                            num_iters=opt.num_smplify_iters,
                            device=device)

        start = time.time()
        for idx in range(num_pers):
            for frm in range(seq_len):
                keypoints_3d[idx*seq_len+frm, :, :] = torch.Tensor(data[idx][frm]).to(device).float()
                pred_betas[idx*seq_len+frm, :] = init_mean_shape
                pred_pose[idx*seq_len+frm, :] = init_mean_pose
                pred_cam_t[idx*seq_len+frm, :] = cam_trans_zero


        # ----- from initial to fitting -------
        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = smplify(pred_pose.detach(),
                                                    pred_betas.detach(),
                                                    pred_cam_t.detach(),
                                                    keypoints_3d,
                                                    conf_3d=confidence_input.to(device),
                                                    seq_ind=0)
        out_meshes = smplmodel(betas=new_opt_betas, global_orient=new_opt_pose[:, :3], body_pose=new_opt_pose[:, 3:],
                                transl=new_opt_cam_t, return_verts=True)
        out_meshes = out_meshes['vertices'].cpu().detach().numpy()
        out_mesh_1, out_mesh_2 = out_meshes[:seq_len], out_meshes[seq_len:]
        print('{:.2f} second'.format(time.time()-start))
        
        with open(out_name, 'wb') as f:
            pickle.dump([out_mesh_1,out_mesh_2], f)
    # render results
    background = np.zeros((opt.height, opt.width, 3))
    renderer = get_renderer(opt.width, opt.height)
    render_video(out_mesh_1, out_mesh_2, renderer, dir_save, background)
    