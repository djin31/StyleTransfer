#!/usr/bin/env python

# ## Neural Algorithm of Artistic Style
# 
# Pytorch implementation of style transfer technique proposed by [Gatys et al, 2016](https://arxiv.org/abs/1508.06576).

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import argparse
import datetime

parser = argparse.ArgumentParser(description='Neural Algorithm of Artistic Style')
parser.add_argument('--content-path', type=str, required=True, help='path to content image')
parser.add_argument('--style-path', type=str, required=True, help='path to style image')
parser.add_argument('--output', type=str, default="examples/", required=False, help='output directory path')
parser.add_argument('--content-layers', type=int, nargs='+', default=[4], required=False, help='list of numbers indicating convolutional layers to use for content activations')
parser.add_argument('--style-layers', type=int, nargs='+', default=[1,2,3,4,5], required=False, help='list of numbers indicating convolutional layers to use for style activations')
parser.add_argument('--content-weight', type=float, default=1.0, required=False, help='factor controlling contribution of content loss to total loss')
parser.add_argument('--style-weight', type=float, default=100.0, required=False, help='factor controlling contribution of style loss to total loss')
parser.add_argument('--learning-rate', type=float, default=1.0, required=False)
parser.add_argument('--num-steps',type=int, default=400, required=False, help="number of steps to be taken by optimizer")
parser.add_argument('--image-size', type=int, default=224, required=False, help="dimension of image")
parser.add_argument('--use-gpu', action='store_true' , help="use gpu for computation")


class EmbeddingGenerator(nn.Module):
    '''
    Compute activations in content and style layers using pretrained VGG19 model
    '''
    def __init__(self, content_layers=[4], style_layers=[1,2,3,4,5]):
        '''
        Input:
            content_layers: list of numbers indicating convolutional layers to use for content activations
            style_layers: list of numbers indicating convolutional layers to use for style activations
        '''
        super(EmbeddingGenerator, self).__init__()
        
        # load pretrained model and set requires grad to be false for each layer
        pretrained_model = models.vgg19(pretrained=True).features.eval().to(device)
        for layer in pretrained_model:
            layer.requires_grad = False
         
        # extract layers uptil the last of content and style layers
        self.pretrained_layers = []
        self.content_layers = []
        self.style_layers = []
        required_layers = max(max(content_layers), max(style_layers))
        conv_counter = 0
        for i,layer in enumerate(pretrained_model):
            if isinstance(layer,nn.Conv2d):
                conv_counter+=1
                if conv_counter>required_layers:
                    break
                if conv_counter in content_layers:
                    self.content_layers.append(i)
                if conv_counter in style_layers:
                    self.style_layers.append(i)
            
            self.pretrained_layers.append(layer)
        
        self.content_layers = set(self.content_layers)
        self.style_layers = set(self.style_layers)
        
        
    def forward(self, x):
        '''
        Input:
            x - Tensor of shape (1 x 3 x H X W)
        Returns:
            content_activations: list of flattened activations at content layers
            style_activations: list of activations at style layers with shape (n_C X (H*W))
        '''
        content_activations = []
        style_activations = []
        
        for i,layer in enumerate(self.pretrained_layers):
            x = layer(x)
            if i in self.content_layers:
                content_activations.append(x.flatten())
            
            if i in self.style_layers:
                style_activations.append(x.view(x.shape[1],-1))
        
        return content_activations, style_activations


class StyleGenerator:
    '''
    Trainer class that generates output image
    '''
    def __init__(self, embedding_gen):
        '''
        Input:
            embedding_gen: object of EmbeddingGenerator class
        '''
        self.model = embedding_gen
        
        # for normalizing input content and style images
        self.preprocess = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        
        # for denormalizing output image
        self.postprocess = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                               ])
        
    def gram_matrix(self,x):
        '''
        Compute Gram matrix for given 2D matrix
        '''
        return torch.matmul(x,x.transpose(0,1))/x.numel()

    def compute_content_loss(self,content_activations,inp_content_activations):
        '''
        Inputs:
            content_activations: content layer activations for content image
            inp_content_activations: content layer activations for generated image
        Returns:
            content_loss
        '''
        content_loss = 0.0
        for cont_activ, inp_cont_activ in zip(content_activations, inp_content_activations):
            diff = cont_activ-inp_cont_activ
            content_loss += torch.sum(torch.square(diff))/diff.numel()
        
        return content_loss
    
    def compute_style_loss(self, style_activations, inp_style_activations):
        '''
        Inputs:
            style_activations: style layer activations for style image
            inp_style_activations: style layer activations for generated image
        Returns:
            style_loss
        '''
        style_loss = 0.0
        for style_activ, inp_style_activ in zip(style_activations, inp_style_activations):
            diff = style_activ-inp_style_activ
            style_loss += torch.sum(torch.square(diff))
        
        return style_loss
    
    def generate_stylized_image(self, content_path, style_path, output_dir="images/",
                                content_wt=1.0, style_wt=1000.0, learning_rate=1.0, num_steps=400, image_size=(224,224)):
        '''
        Inputs:
            content_path: path to content image
            style_path: path to style image
            output_dir: output directory for intermediate generated images
            
            content_wt: factor controlling contribution of content loss to total loss
            style_wt: factor controlling contribution of style loss to total loss
            
            learning_rate: learning rate of optimizer
            num_steps: number of steps to be taken by optimizer
            
            image_size: tuple indicating size of input and output images to be used, default is input size expected by VGG19
        '''
        start_time = datetime.datetime.now()
        # create output directory
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        print("Saving output images to {}".format(output_dir))
        
        # load content and style images and compute relevant activations
        content_img = Image.open(content_path).resize(image_size)
        style_img = Image.open(style_path).resize(image_size)
        
        content = self.preprocess(content_img).unsqueeze(0).to(device)
        style = self.preprocess(style_img).unsqueeze(0).to(device)
        
        content_activations = [x.detach() for x in self.model(content)[0]]   
        style_activations = [self.gram_matrix(x.detach()) for x in self.model(style)[1]]

        # initialize output image with content image + noise, results in faster convergence
        gaussian_noise = torch.clamp(torch.randn(1,3,image_size[0], image_size[1]),-1,1)*0.5
        gen_image = content*0.5 + gaussian_noise.to(device)*0.5
        gen_image = nn.Parameter(gen_image)
        
        # initialize optimizer with gen_image as parameters over which optimization is carried out
        optimizer = torch.optim.LBFGS([gen_image.requires_grad_()], lr=learning_rate)
        
        # initialize as list to allow pass by reference in closure function
        runs=[0]
        while runs[0]<num_steps:
            def closure():
                '''
                closure function required by LBFGS optimizer
                '''
                optimizer.zero_grad()
                
                inp_content, inp_style = self.model(gen_image)
                inp_style = [self.gram_matrix(x) for x in inp_style]

                content_loss = self.compute_content_loss(content_activations, inp_content)
                style_loss = self.compute_style_loss(style_activations, inp_style)
                loss = content_wt*content_loss + style_wt*style_loss
                runs[0]+=1
                if runs[0]%40==0:
                    print("Num Steps: {} \tContent Loss: {} \tStyle Loss: {} \tTotal Loss:{}".format(runs[0], 
                                round(content_loss.item(),3), round(style_loss.item(),3), round(loss.item(),3)))
                    # save intermediate outputs
                    plt.imsave(os.path.join(output_dir,"epoch_"+str(runs[0])+".jpg"), 
                               torch.clamp(self.postprocess(gen_image[0].cpu().detach()).permute(1,2,0),0,1).numpy())

                loss.backward()
                return loss
            optimizer.step(closure)
                
        
        # save final image 
        fig,ax = plt.subplots(1,3, figsize=(15,5),facecolor='w')
        ax[0].imshow(content_img)
        ax[0].set_title("Content")
        ax[1].imshow(style_img)
        ax[1].set_title("Style")
        ax[2].imshow(torch.clamp(self.postprocess(gen_image[0].cpu().detach()).permute(1,2,0),0,1).numpy())
        ax[2].set_title("Generated")
        fig.savefig(os.path.join(output_dir,"final.jpg"))

        print("Time taken {}".format(datetime.datetime.now()-start_time))


# #### Important implementation details 
# * Use LBFGS as optimizer. Adam or SGD optimizer do not give sharp outputs.
# * Remember to normalize input content and style images using mean and std deviation required by VGG19
# * Remember to clamping output values and denormalize output image.
# * Keep ratio of style_wt to content_wt more than 10, preferably around 100. 







if __name__=="__main__": 
    args = parser.parse_args()
    device="cpu"
    if torch.cuda.is_available() and args.use_gpu:
        device="cuda"
    print("Using device: {}".format(device))

    embed = EmbeddingGenerator(content_layers=args.content_layers, style_layers=args.style_layers)
    print("Loaded pretrained model")
    styler = StyleGenerator(embed)
    styler.generate_stylized_image(content_path=args.content_path,style_path=args.style_path, output_dir=args.output, 
                            content_wt=args.content_weight, style_wt=args.style_weight, 
                            learning_rate=args.learning_rate, num_steps=args.num_steps,
                            image_size=(args.image_size,args.image_size))
