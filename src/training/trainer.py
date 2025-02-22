from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy


class TEXTure:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        # JA: TEXTure uses loguru for custom logs (i.e. alternative to using print() for logs).
        # The self.init_logger() function simply sets up this custom logging framework.
        self.init_logger()

        # JA: Pyrallis is an alternative to arg parse.
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        # JA: When requested during generation, images generated with specified view directions
        # can be used for a specific side of the surface. This is necessary for when one side of
        # an object has a unique appearance from that of another surface (e.g. a human being)
        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom']
        self.mesh_model = self.init_mesh_model()
        self.diffusion = self.init_diffusion()

        if self.cfg.guide.embeds_path is not None:
            self.diffusion.load_textual_inversion(self.cfg.guide.embeds_path)
            logger.info(f'Loaded textual inversion for {self.cfg.guide.embeds_path}')

        #breakpoint()
        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0  #MJ: brick wall image

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_mesh_model(self) -> nn.Module:
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> Any:
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          concept_path=self.cfg.guide.concept_path,
                                          latent_mode=False,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
            text_string = ref_text
        else:
            text_z = []
            text_string = []

            for d in self.view_dirs:
                negative_prompt = None
                if d == 'front' and self.cfg.guide.text_front:
                    text = self.cfg.guide.text_front
                elif d == 'back' and self.cfg.guide.text_back:
                    text = self.cfg.guide.text_back
                elif d == 'left' and self.cfg.guide.text_left:
                    text = self.cfg.guide.text_left
                elif d == 'right' and self.cfg.guide.text_right:
                    text = self.cfg.guide.text_right
                elif d == 'overhead' and self.cfg.guide.text_overhead:
                    text = self.cfg.guide.text_overhead
                elif d == 'bottom' and self.cfg.guide.text_bottom:
                    text = self.cfg.guide.text_bottom
                else:
                    text = ref_text.format(d)

                text_string.append(text)
                logger.info(text)
                logger.info(negative_prompt)

                text_z.append(self.diffusion.get_text_embeds(
                    [text],
                    negative_prompt=negative_prompt,
                    append_direction=self.cfg.guide.append_direction,
                    dir_embed_factor=self.cfg.guide.dir_embed_factor
                ))
        return text_z, text_string

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        # JA: self.cfg.render is a subcategory of config called RenderConfig
        init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()

        # JA: eval_size is the number of angles to sample after training and is set to 10
        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video

        # JA: full_eval_size is the number of angles to sample after training and is set to 100
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # JA: Somewhat counterintuitive to the way we normally think of "dataset" or "dataloader" the train dataset and dataloader
        # simply refer to the properties of every angle.
        #=>MJ: But in fact, dataset plays the same role as the usual dataset; 
        # The only difference is that the target/label of each input camera view in the dataset is obtained from the image generated
        # by the stable diffusion pipeline, which generates images from the text prompt, the depth, and the reference image for inpainting.
        
        #MJ: The following loop is the "batch loop" which takes each batch (with batch size =1) from the dataloader;
        #MJ: It also uses only one epoch; It loops over the dataset only once. But we would need to scan over the dataset several times
        #    using different orders of view; The learning of the texture atlas is different depending on this order, so we need to get the
        # avergage result from using many different orders. 
        
        #TODO: MJ: Try to add an epoch loop: try different num of epochs.
        
        for i, data in enumerate(self.dataloaders['train']):
            #breakpoint()
            logger.info(f"Data at iteration {i}: {data}")
            self.paint_step += 1
            self.trace_number = 0
            pbar.update(1)
            self.paint_viewpoint(data)
            self.evaluate(self.dataloaders['val'], self.eval_renders_path)
            self.mesh_model.train()

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            
            preds, textures, depths, normals = self.eval_render(data) 
            #MJ: preds = the predicted image_features rendered from the face_features defined over each vertex on each face on self.mesh_model

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'{self.paint_step:04d}_{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.paint_step:05d}_{name}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    def paint_viewpoint(self, data: Dict[str, Any]):
        #breakpoint()
        # JA: The main code used for training of the texture altas from a single view in the training dataset.
        # data represents the information about a single view in the training dataset (MultiviewDataset).
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius'] # JA: Values are in radians
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)    # JA: front_offset is set to 0.0 by default.
                                                                # Thus, this is effectively just phi = phi - 0.0

        # JA: If phi is negative (e.g. -10 deg), rewrite it to be positive (e.g. 350 deg)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)

        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if self.cfg.guide.use_background_color: # JA: use_background_color is set to False by default.
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else: #MJ: this is the case
            background = F.interpolate(self.back_im.unsqueeze(0), #MJ: brick wall image
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background) #MJ background=brick wall img
        render_cache = outputs['render_cache'] # 
        rgb_render_raw = outputs['image']  # Render where missing values have special color;
        #MJ: used in self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                            # JA: rgb_render_raw is the Q_t mentioned in the paper
        depth_render = outputs['depth'] # JA: depth_render is the D_t in the paper

        # JA: the render function does an assert function to check that EITHER [theta, phi, radius] OR [render_cache]
        # is passed to it.
        # The  render function is called twice, with different input parameters set.
        # In the second call: If self.paint_step is > 1 (remember that self.paint_step ranges from 0 to
        # 7==> MJ: BO. paint_step ranges from 1 to 50, the sampling steps of stable diffusion), use_median is set to True.
        # FIND OUT LATER: But what is use_median and why is it being used here?

        # Render again **with the median value to use as rgb**, we shouldn't have color leakage, but just in case
        #MJ: "Color leakage" refers to a phenomenon in which the color from one object or area of an image spills
        # or bleeds into adjacent regions, resulting in an unwanted blending or mixing of colors. 
        # It is a common issue encountered in digital imaging and computer graphics.
        # Addressing color leakage often involves careful color correction, edge refinement, and the use of masking techniques to separate and blend different elements more effectively. Advanced image editing software provides tools and
        # filters specifically designed to mitigate color leakage and improve overall color accuracy and fidelity.
        # JA: This second rendering from the texture map is to avoid the potential color leakage but this is optional
        outputs = self.mesh_model.render(background=background, #MJ: background=brick wall img
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        texture_map = outputs['texture_map'] # JA: texture_map is the texture_img because the above render function is called with use_meta_texture == False

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1) #MJ: z_normals was computed by the previous render

        # JA: From the paper:
        # To keep track of seen regions and the cross-section at which they were previously colored from, we use an
        # additional meta-texture map N that is updated at every view iteration of rendering. This additional map can be ef- ficiently
        # rendered together with the texture map at each view iteration and is used to define the current trimap
        # partitioning.

        # Render meta texture map # JA: This additional map is used for the trimap:
        # MJ: It uses self.meta_texture to produce the image_features, and the meta_texutre is learned
        # to match the z_normals of the faces in the camera coordinate system.
        #Note: meta_output: 
        #  {'image': pred_map, 'mask': mask, 'background': pred_back,
        #         'foreground': pred_features, 'depth': depth, 'normals': normals, 'render_cache': render_cache,
        #         'texture_map': texture_img}
        # Here the value of "image" key = the image_features mentioned above.
        # JA: Render from meta texture which contains the z normals of the previous views
        # meta textures is set to 0 initially but as the viewpoints accumulate, it stores the z normal caches of all the previous camera views
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device), #MJ: rendering of meta-texture using black background
                                             use_meta_texture=True, render_cache=render_cache)

        # z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1) #MJ: z_normals was computed by the previous render

        z_normals_cache_of_previous_view = meta_output['image'].clamp(0, 1)  # JA: z_normals_cache represents the meta texture map
                                                            # (z normals of the vertices in the previous camera view)

        #MJ: z_normals_cache: the rendered image of the meta-texutre, which is Parameter to be learned and
        # is set to zero initially.
        # z_normals_cache contrains the z_normals of the previous camera view, obtained from the meta_texture;
        # For the first view, z_normals_cache is zero, because meta_texture is set to zero initially:
        # self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img))
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2] 
        #MJ:: shape = (1,1,1200,1200); z_normals_cache: shape=(1,3,1200,1200); meta_output['image'].clamp(0, 1)[:, 1:2] is used
        # for specific purpose in calculate_tripmap(). But either z_normals_cache or edited_mask is used dependng on the cases

        #MJ meta_texture_img = self.mesh_model.meta_texture_img[0,0]
        meta_texture_img = meta_output['texture_map'] #MJ:  texture_map is meta_texture_img because the above render function is called with use_meta_texture == True
        self.log_train_image(rgb_render, 'rgb_render(Q_t)')
        self.log_train_image(depth_render[0, 0], 'depth(D_t)', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals(z_of_normal_vectors_in_curr_view)', colormap=True)
        self.log_train_image(z_normals_cache_of_previous_view[0, 0], 'z_normals_cache_of_previous_views_seen_in_curr_view', colormap=True)
        self.log_train_image(texture_map, 'color_texture_map')
        self.log_train_image(meta_texture_img[0,0], 'z_normals_cache_of_all_previous_views', colormap=True) #MJ:  we log the first channel of meta_texture_img with pseudo colormap

        # text embeddings
        if self.cfg.guide.append_direction:
            # JA: append_direction is set to True in the configs

            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}') # JA: text_string is only used here to log, and not used anywhere else.
                                            # JA: However text_z is used underneath when calling stable diffusion.

        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache_of_previous_view,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'])

        # JA: All values in update_mask are either 0 or 1
        # update_mask.shape[2] * update_mask.shape[3] = 1440000
        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return

        self.log_train_image(rgb_render * (1 - update_mask), name='masked_input(region_not_to_be_updated)') #MJ: The area to be updated will be zero, black region
        self.log_train_image(rgb_render * update_mask, name='masked_input(region_to_be_updated)') #MJ: The area to be updated will be zero, black region
        self.log_train_image(rgb_render * refine_mask, name='region_to_be_refined')

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render)
        cropped_depth_render = crop(depth_render)
        cropped_update_mask = crop(update_mask)
        self.log_train_image(cropped_rgb_render, name='cropped_rgb_render_for_sd')

        checker_mask = None
        if self.paint_step > 1:
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (512, 512)) * (1 - checker_mask),
                                 'checkerboard_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1

        experiment_number = 1

        if experiment_number == 1:
            self.diffusion.use_inpaint = True
            cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(), #MJ: cropped_rgb_render=Q_t=>latents
                                                                        cropped_depth_render.detach(),
                                                                        guidance_scale=self.cfg.guide.guidance_scale,
                                                                        strength=1.0, update_mask=cropped_update_mask,
                                                                        fixed_seed=self.cfg.optim.seed,
                                                                        check_mask=checker_mask,
                                                                        intermediate_vis=self.cfg.log.vis_diffusion_steps)
        elif experiment_number == 2:
            self.diffusion.use_inpaint = True
            #MJ: experiment 2: try to use img2img_step with setting update_mask and without setting check_mask:
            cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(), #MJ: cropped_rgb_render=Q_t=latents
                                                                        cropped_depth_render.detach(),
                                                                        guidance_scale=self.cfg.guide.guidance_scale,
                                                                        strength=1.0, update_mask=cropped_update_mask,
                                                                        fixed_seed=self.cfg.optim.seed,
                                                                        check_mask=None,
                                                                        intermediate_vis=self.cfg.log.vis_diffusion_steps)
        elif experiment_number == 3:
            self.diffusion.use_inpaint = False
            #MJ: experiment 3: try to use img2img_step without setting update_mask and check_mask: 
            # In this case, you should set self.diffusion.use_inpainting=False by setting self.cfg.guide.use_inpainting to false
            cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(), #MJ: cropped_rgb_render=Q_t=latents
                                                                        cropped_depth_render.detach(),
                                                                        guidance_scale=self.cfg.guide.guidance_scale,
                                                                        strength=1.0, update_mask=None,
                                                                        fixed_seed=self.cfg.optim.seed,
                                                                        check_mask=None,
                                                                        intermediate_vis=self.cfg.log.vis_diffusion_steps)
        else:
            raise NotImplementedError
            
            
        # JA: cropped_rgb_output is the stable diffusion-generated image from the prompt text_z
        self.log_train_image(cropped_rgb_output, name='cropped_rgb_output_from_sd')
        self.log_diffusion_steps(steps_vis)

        # JA: From https://velog.io/@pindum/PyTorch-interpolation
        # interpolation이 무엇인가 하면 사전적으로는 보간이라는 뜻을 가지며 작은 사이즈의 이미지를 큰 사이즈로 키울 때 사용된다.
        # 단순히 업샘플링이라고 할 수도 있지만 늘어날 때 중간 값을 적절하게 보간해주는 옵션들을 구체적으로 구현하고 있다.
        cropped_rgb_output = F.interpolate(cropped_rgb_output,
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        rgb_output = rgb_render.clone()
        # JA: (min_h, max_h) and (min_w, max_w) define the bounding box of the object region
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output
        self.log_train_image(rgb_output, name='sd_output_within_background')

        # Project back
        object_mask = outputs['mask']
        #breakpoint()
        fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               z_normals_cache=z_normals_cache_of_previous_view)
        self.log_train_image(fitted_pred_rgb, name='rendered_image_from_learned_texture_atlas')

        return

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask

        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache'])

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
        pred_z_normals = meta_output['image'][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals

    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor):
        #MJ: Note that self.texture_img was set to the default color initially to make sense of the
        # following two lines.
        
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0) # JA: exact_generate_mask.shape = [1, 1, 1200, 1200]
        #MJ: The area of the rendered image, Q_{t} that are close to the default purple color is the candidate to be 
        #  generated by the current view of the camera. The texture_img from which 
        # rgb_render_raw was generated was set to this default color initially. The fact that some area of rgb_rendered_raw 
        # have the default color means that this area was NOT rendered before.

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0) # JA: exact_generate_mask[0, 0].shape = [1200, 1200]

        update_mask = generate_mask.clone()  #MJ: generate_mask is used to define update_mask

        object_mask = torch.ones_like(update_mask)  ##MJ: how is this object_mask different from 'mask' passed as parameter
        object_mask[depth_render == 0] = 0 # JA: depth_render is the D_t of the previous view, D_t = 0 means the area with no object
        object_mask = torch.from_numpy( # JA: https://nicewoong.github.io/development/2018/01/05/erosion-and-dilation/
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask: The region where the z component of face normal is big enough compared to those obtained in the previous camera view is to be refined. To store the information about the previous camera view, the meta-texture is used!!!

        refine_mask = torch.zeros_like(update_mask) #MJ: z_normals_cache: the rendered image of the meta-texutre:z_normals_cache: shape=(1,3,1200,1200), 3= 3 vertices of the face
        refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1 # JA: z_update_thr is 0.2; MJ: edited_mask= z_normals_cache[:, :1, :, :]
        if self.cfg.guide.initial_texture is None: #MJ: this is the case
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0  #MJ: edited_mask: N/A; How would using edited_mask be different from using z_normals_cache[:, :1, :, :]
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)  #MJ: mask: N/A
            refine_mask[mask == 0] = 0
            # Don't use bad angles here
            refine_mask[z_normals < 0.4] = 0 # JA: If the z-normal of the triangle is less than 0.4, the direction of the face normal is bad with respect to the camera direction
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0

        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1  #MJ: refine_mask is used to define update_mask.

        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0 #MJ: generate_mask is used to define update_mask

        # Visualize trimap
        if self.cfg.log.log_images: #MJ: this is the case
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - exact_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * exact_generate_mask

            if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            
            self.log_train_image(generate_mask[0,0], 'generate_mask', colormap=True)
            self.log_train_image(refine_mask[0,0], 'refine_mask', colormap=True)
            self.log_train_image(update_mask[0,0], 'update_mask', colormap=True)
            self.log_train_image(shaded_rgb_vis, 'rgb_render_raw_shaded_with_white')
            self.log_train_image(trimap_vis, 'trimap')

        return update_mask, generate_mask, refine_mask
    # checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask), crop(generate_mask))
    def generate_checkerboard(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(checkerboard,
                                     (512, 512))
        checker_mask = F.interpolate(update_mask_inner, (512, 512))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1, #MJ: the area to be refined
                                                        update_mask_base_inner == 0).float(), (512, 512)) #MJ: the area not to be generated
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1] # replace the checkerboard mask into the area to be refined
        return checker_mask

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):  # JA: rgb_output is I_t
        #breakpoint()
        object_mask = torch.from_numpy( 
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        render_update_mask = object_mask.clone() # JA: We will update the object part only

        render_update_mask[update_mask == 0] = 0 # JA: within the object region, we will not update where object_mask is 0

        blurred_render_update_mask = torch.from_numpy(
            cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
            render_update_mask.device).unsqueeze(0).unsqueeze(0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals # JA: z-normals represents the z-component of the face normals for each pixel
            z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :] # JA: z_update_thr is 0.2
            # z_was_better ==  self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :] - z_normals # JA: z_update_thr is 0.2
            blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask # JA: Set the final render_update_mask
        self.log_train_image(rgb_output * render_update_mask, 'project_back_input(train_target)')

        # Update the normals
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])

        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            # JA: rgb_output is I_t
            # 2D tensor is converted to 1D
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean() + (
                    (masked_pred - masked_pred.detach()).pow(2) * (1 - masked_mask)).mean() # JA: (masked_pred - masked_pred.detach()) should be meaningless

            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            current_z_mask = meta_outputs['mask'].flatten()  #MJ: the object mask of the mesh
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :, #MJ: current_z_normals.shape[1]=3
                                       current_z_mask == 1][:, :1] #MJ: Select the first channel of current_z_normals, whose value =z_normals_cache[:, 0, :, :] 
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            
            loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            
            loss.backward()
            optimizer.step()

        return rgb_render, current_z_normals

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            self.trace_number += 1
            # JA: Create the file names so that they reveal the order in which they are logged
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'{self.paint_step:04d}_{self.trace_number:02d}_{name}.jpg')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            Image.fromarray(
                (einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)).save(
                path)
