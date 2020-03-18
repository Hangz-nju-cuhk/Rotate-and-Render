import os.path as osp
import numpy as np
import neural_renderer as nr
import torch
import scipy.io as sio
import pickle
import skimage.transform as trans
from math import *
from models.networks.render import *


class TestRender2(Render):

    def __init__(self, opt):
        super(TestRender2, self).__init__(opt)

    def _parse_param(self, param, pose_noise=False, frontal=True):
        """Work for both numpy and tensor"""
        p_ = param[:12].reshape(3, -1)
        p = p_[:, :3]
        s, R, t3d = P2sRt(p_)
        if frontal:
            angle = matrix2angle(R)
            angle[0] = 0
            p = angle2matrix(angle) * s
        if pose_noise:
            angle = matrix2angle(R)
            angle[0] = np.random.uniform(-0.258, 0.258, 1)[0]
            p = angle2matrix(angle) * s

        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = param[12:52].reshape(-1, 1)
        alpha_exp = param[52:-4].reshape(-1, 1)
        box = param[-4:]
        return p, offset, alpha_shp, alpha_exp, box


    def rotate_render(self, params, images, M=None, with_BG=False, random_color=False,
                      pose_noise=False, align=True, frontal=True, erode=True, grey_background=False, avg_BG=True):

        bz, c, w, h = images.size()


        face_size = self.faces.size()
        self.faces = self.faces.expand(bz, face_size[1], face_size[2])

        # get render color vertices and normal vertices information, get original texs
        vertices = []
        vertices_out = []
        vertices_in_ori_img = []
        vertices_aligned_normal = []
        vertices_aligned_out = []
        vertices_ori_normal = []
        texs = []

        for n in range(bz):
            tex_a, vertice, vertice_out, vertice_in_ori_img, align_vertice = self._forward(params[n], images[n], M[n], random_color,
                                                        pose_noise=pose_noise, align=align, frontal=frontal)
            vertices.append(vertice)
            vertices_out.append(vertice_out)
            vertices_in_ori_img.append(vertice_in_ori_img.clone())
            vertice2 = self.flip_normalize_vertices(vertice_in_ori_img.clone())
            vertices_ori_normal.append(vertice2)
            vertices_aligned_out.append(align_vertice)
            align_vertice_normal = self.flip_normalize_vertices(align_vertice.clone())
            vertices_aligned_normal.append(align_vertice_normal.clone())
            texs.append(tex_a)

        vertices = torch.cat(vertices, 0)
        vertices_aligned_normal = torch.cat(vertices_aligned_normal, 0)
        vertices_ori_normal = torch.cat(vertices_ori_normal, 0)

        vertices_in_ori_img = torch.stack(vertices_in_ori_img, 0)
        vertices_aligned_out = torch.stack(vertices_aligned_out, 0)

        texs = torch.cat(texs, 0)

        # erode the original mask and render again
        rendered_images_erode = None
        if erode:
            with torch.no_grad():
                rendered_images, depths, masks, = self.renderer(vertices_ori_normal, self.faces, texs)  # rendered_images: batch * 3 * h * w, masks: batch * h * w
            masks_erode = self.generate_erode_mask(masks, kernal_size=9)
            rendered_images = rendered_images.cpu()
            if grey_background:
                rendered_images_erode = masks_erode * rendered_images
            else:

                inv_masks_erode = (torch.ones_like(masks_erode) - (masks_erode)).float()
                if avg_BG:
                    contentsum = torch.sum(torch.sum(masks_erode * rendered_images, 3), 2)
                    sumsum = torch.sum(torch.sum(masks_erode, 3), 2)
                    contentsum[contentsum == 0] = 0.5
                    sumsum[sumsum == 0] = 1
                    masked_sum = contentsum / sumsum
                    masked_BG = masked_sum.unsqueeze(2).unsqueeze(3).expand(rendered_images.size())
                else:
                    masked_BG = 0.5
                rendered_images_erode = masks_erode * rendered_images + inv_masks_erode * masked_BG

            texs_a_crop = []
            for n in range(bz):
                tex_a_crop = self.get_render_from_vertices(rendered_images_erode[n], vertices_in_ori_img[n])
                texs_a_crop.append(tex_a_crop)
            texs = torch.cat(texs_a_crop, 0)

        # render face to rotated pose
        with torch.no_grad():
            rendered_images, depths, masks, = self.renderer(vertices, self.faces, texs)

        # add mask to rotated
        masks_erode = self.generate_erode_mask(masks, kernal_size=5)
        inv_masks_erode = (torch.ones_like(masks_erode) - masks_erode).float()
        rendered_images = rendered_images.cpu()
        if with_BG:
            images = torch.nn.functional.interpolate(images, size=(self.render_size))
            rendered_images = masks_erode * rendered_images + inv_masks_erode * images  # 3 * h * w
        else:
            if grey_background:
                if np.random.randint(0, 4):
                    rendered_images = masks_erode * rendered_images
            else:
                if avg_BG:
                    contentsum = torch.sum(torch.sum(masks_erode * rendered_images, 3), 2)
                    sumsum = torch.sum(torch.sum(masks_erode, 3), 2)
                    contentsum[contentsum == 0] = 0.5
                    sumsum[sumsum == 0] = 1
                    masked_sum = contentsum / sumsum
                    masked_BG = masked_sum.unsqueeze(2).unsqueeze(3).expand(rendered_images.size())
                else:
                    masked_BG = 0.5
                rendered_images = masks_erode * rendered_images + inv_masks_erode * masked_BG

        # get rendered face vertices
        texs_b = []
        for n in range(bz):
            tex_b = self.get_render_from_vertices(rendered_images[n], vertices_out[n])
            texs_b.append(tex_b)
        texs_b = torch.cat(texs_b, 0)

        # render back
        with torch.no_grad():
            rendered_images_rotate, depths1, masks1, = self.renderer(vertices_ori_normal, self.faces, texs_b)  # rendered_images: batch * 3 * h * w, masks: batch * h * w
            rendered_images_double, depths2, masks2, = self.renderer(vertices_aligned_normal, self.faces, texs_b)  # rendered_images: batch * 3 * h * w, masks: batch * h * w


        masks2 = masks2.unsqueeze(1)
        inv_masks2 = (torch.ones_like(masks2) - masks2).float().cpu()
        BG = inv_masks2 * images
        if grey_background:
            masks1 = masks1.unsqueeze(1)

            inv_masks1 = (torch.ones_like(masks1) - masks1).float()

            rendered_images_rotate = (inv_masks1 * 0.5 + rendered_images_rotate).clamp(0, 1)
            rendered_images_double = (inv_masks2 * 0.5 + rendered_images_double).clamp(0, 1)


        return rendered_images_rotate, rendered_images_double, \
               self.torch_get_68_points(vertices_in_ori_img), self.torch_get_68_points(vertices_aligned_out), rendered_images_erode, BG
