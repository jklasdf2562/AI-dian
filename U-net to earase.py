import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image
import os
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import cv2  
import matplotlib.pyplot as plt
import random
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#准备数据
def collect_data_pair(data_dir):
    pairs=[]
    for root in data_dir:
        input_dir=os.path.join(root,'dataset','input')
        output_dir=os.path.join(root,'dataset','output')
        input_images=os.listdir(input_dir)
        for img in input_images:
            input_path=os.path.join(input_dir,img)
            output_path=os.path.join(output_dir,img)
            if os.path.exists(output_path):
                pairs.append((input_path,output_path))
            else:
                print(f"Warning: Output image not found for {input_path}")
    return pairs
script_dir=os.path.dirname(os.path.abspath(__file__))
data_dir=[os.path.join(script_dir,'deli',root) for root in ['20250211','20250212','20250213']]
data_pairs=collect_data_pair(data_dir)
print(f"Total data pairs collected: {len(data_pairs)}")
train,val=train_test_split(data_pairs,test_size=0.2,random_state=42)




class ImageDataset(Dataset):
    def __init__(self, data_pairs, input_size=512, mean=None, std=None, is_train=True):
        self.data_pairs = data_pairs
        self.input_size = input_size
        self.mean = mean          # 训练集输入统计量
        self.std = std
        self.is_train = is_train

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_path, output_path = self.data_pairs[idx]
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')
        

        # 调整大小
        input_img = F.resize(input_img, (self.input_size, self.input_size))
        output_img = F.resize(output_img, (self.input_size, self.input_size))

        # 训练时随机水平翻转（同时翻转输入和输出）
        if self.is_train:  
            if random.random() < 0.5:
                input_img = F.hflip(input_img)
                output_img = F.hflip(output_img)
            if random.random() < 0.5:
                input_img = F.vflip(input_img)
                output_img = F.vflip(output_img)
            if random.random() < 0.5:
                angle=random.uniform(-5, 5)  # 随机旋转角度
                input_img = F.rotate(input_img, angle, expand=False, fill=0)
                output_img = F.rotate(output_img, angle, expand=False, fill=0)

            if random.random()<0.3:
                brightness_factor = random.uniform(0.9, 1.1)
                input_img = F.adjust_brightness(input_img, brightness_factor)
            if random.random()<0.3:
                contrast_factor = random.uniform(0.9, 1.1)
                input_img = F.adjust_contrast(input_img, contrast_factor)
                

        # 转换为张量，数值范围 [0,1]
        input_tensor = F.to_tensor(input_img)   # (C, H, W), float32, [0,1]
        output_tensor = F.to_tensor(output_img) 
        if self.is_train and random.random() < 0.2:
            noise = torch.randn_like(input_tensor) * 0.02  # 标准差 0.02
            input_tensor = input_tensor + noise
            input_tensor = torch.clamp(input_tensor, 0.0, 1.0)

        # 对输入进行归一化（标准化）
        if self.mean is not None and self.std is not None:
            input_tensor = F.normalize(input_tensor, mean=self.mean, std=self.std)

        return input_tensor, output_tensor
#数据预处理
def compute_mean_std(data_pairs):
    channel_sum=np.zeros(3)
    channel_squared_sum=np.zeros(3)
    num_pixels=0
    channel_sum_output=np.zeros(3)
    channel_squared_sum_output=np.zeros(3)
    num_pixels_output=0
    for input_path,output_path in tqdm(data_pairs,desc="Computing mean and std"):
        image=np.array(Image.open(input_path).convert('RGB'))/255.0
        channel_sum+=image.sum(axis=(0,1))
        channel_squared_sum+=(image**2).sum(axis=(0,1))
        num_pixels+=image.shape[0]*image.shape[1]

    mean_input=channel_sum/num_pixels
    std_input=np.sqrt((channel_squared_sum/num_pixels-mean_input**2))

    print(f"Computed mean: {mean_input}, std: {std_input}")
    return mean_input,std_input


means,stds=compute_mean_std(train)


train_dataset = ImageDataset(train, input_size=512, mean=means, std=stds, is_train=True)
val_dataset = ImageDataset(val, input_size=512, mean=means, std=stds, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


#定义模型
class cnn_block(nn.Module):
    def __init__(self,in_chnnels,out_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_chnnels,out_channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.act2=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.act1(x)
            x=self.conv2(x)
            x=self.bn2(x)
            x=self.act2(x)
            skip=x
            x=self.pool(x)
            return x,skip
class encoder(nn.Module):
    def __init__(self,in_channels=3,embed_dim=1024): #[B,3,512,512]->[B,512,32,32]
        super().__init__()
        self.inchannels=in_channels
        self.channels=[(in_channels,64),(64,128),(128,256),(256,512)]
        self.blocks=nn.ModuleList([cnn_block(in_c,out_c) for in_c,out_c in self.channels])
        self.final_conv=nn.Conv2d(512,embed_dim,kernel_size=3,padding=1)
        self.act=nn.ReLU(inplace=True)
        self.final_conv2=nn.Conv2d(embed_dim,embed_dim,kernel_size=3,padding=1)
        self.act2=nn.ReLU(inplace=True)
    def forward(self,x):
        skips=[]
        for block in self.blocks:
            x,skip=block(x)
            skips.append(skip)
        x=self.final_conv(x)
        x=self.act(x)
        x=self.final_conv2(x)
        x=self.act2(x)
        return x,skips
class upblock(nn.Module):
    def __init__(self,in_channels,out_channels,skip_channels):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.skip_channels=skip_channels
        self.conv_pre=nn.Conv2d(in_channels,in_channels//2,kernel_size=1)
        self.conv1=nn.Conv2d(in_channels//2+skip_channels,out_channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.act2=nn.ReLU(inplace=True)
    def forward(self,x,skip):
        x=self.upsample(x)
        x=self.conv_pre(x)
        x=torch.cat([x,skip],dim=1)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)      
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        return x
class decoder(nn.Module):
    def __init__(self,embed_dim=1024,out_channels=3,final_size=512): #[B,1024,32,32]->[B,3,512,512]
        super().__init__()
        self.final_size=final_size
        self.channels=[(embed_dim,512,512),(512,256,256),(256,128,128),(128,64,64)]
        self.skip_channels = [512,256,128,64]  
        self.upblocks=nn.ModuleList([
            upblock(in_c, out_c, skip_c) for (in_c, out_c, skip_c) in zip([embed_dim,512,256,128],[512,256,128,64],self.skip_channels)
        ])
        self.final_conv=nn.Conv2d(64,out_channels,kernel_size=3,padding=1)
        self.final_act=nn.Sigmoid()
    def forward(self,x,skips):
        for upblock, skip in zip(self.upblocks, skips[::-1]):#skips[64,128,256,512]
            x = upblock(x, skip)
        x = self.final_conv(x)
        x = self.final_act(x)

        return x
class Unet(nn.Module):
    def __init__(self,image_size=512,in_channels=3,embed_dim=512,out_channels=3):
        super().__init__()
        self.encoder=encoder(in_channels,embed_dim)
        self.decoder=decoder(embed_dim,out_channels,image_size)
    def forward(self,x):
        x,skips=self.encoder(x)
        x=self.decoder(x,skips)

        return x


    
#训练模型
def train_model(train_loader,val_loader,model):
    lr=0.001
    wieght_decay=1e-5
    criterion_l1=nn.L1Loss()
    critertion_ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wieght_decay)
    schedeler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)
    best_loss=float('inf')
    start_epoch=0
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        schedeler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming training from epoch {start_epoch}, best_loss={best_loss:.4f}")
    patience_counter=0
    patience=10
    train_losses = []
    val_losses = []
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    def denormalize(tensor, mean, std):
        mean=torch.tensor(mean).view(1,3,1,1).to(tensor.device)
        std=torch.tensor(std).view(1,3,1,1).to(tensor.device)
        return tensor * std + mean
    
    for epoch in range(start_epoch,101): 
        model.train()
        train_loss=0.0
        loop=tqdm(train_loader,desc=f"Epoch {epoch+1}/100")
        for inputs,targets in loop:
            inputs,targets=inputs.to(device),targets.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss_l1=criterion_l1(outputs,targets)  # [B,3,H,W]
                
            loss=loss_l1+0.5*(1-critertion_ssim(outputs,targets))
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()*inputs.size(0)
        train_loss/=len(train_loader.dataset)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for inputs,targets in val_loader:
                
                inputs,targets=inputs.to(device),targets.to(device)
                outputs=model(inputs)
                loss=criterion_l1(outputs,targets)
                val_loss+=loss.item()*inputs.size(0) 
                psnr.update(outputs, targets)
                ssim.update(outputs, targets)   
        val_loss/=len(val_loader.dataset)
        val_losses.append(val_loss)
        avg_psnr = psnr.compute().item()
        avg_ssim = ssim.compute().item()
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val PSNR: {avg_psnr:.4f}, Val SSIM: {avg_ssim:.4f}")
        schedeler.step(val_loss)
        # 每1轮保存2张效果图
        if (epoch+1) % 1 == 0:
            model.eval()
            cnt = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    if inputs.size(0) > 0:
                        indices=random.sample(range(inputs.size(0)), min(2, inputs.size(0)))
                        for i in indices:
                            inp = denormalize(inputs[i], means, stds).squeeze(0)
                            out = outputs[i].squeeze(0)
                            tgt = targets[i].squeeze(0)
                            print('inp shape',inp.shape)
                            inp=inp.cpu().numpy().transpose(1,2,0)
                            out=out.cpu().numpy().transpose(1,2,0)
                            tgt=tgt.cpu().numpy().transpose(1,2,0)
                            inp=(inp*255).astype(np.uint8)
                            out=(out*255).astype(np.uint8)
                            tgt=(tgt*255).astype(np.uint8)
                            fig, axs = plt.subplots(1,3,figsize=(12,4))
                            axs[0].imshow(np.clip(inp,0,255))
                            axs[0].set_title('Input')
                            axs[1].imshow(np.clip(out,0,255))
                            axs[1].set_title('Output')
                            axs[2].imshow(np.clip(tgt,0,255))
                            axs[2].set_title('Target')
                            for ax in axs:
                                ax.axis('off')
                            plt.tight_layout()
                            plt.savefig(f'epoch{epoch+1}_sample{cnt+1}.png')
                            plt.close()
                            cnt += 1
                            if cnt >= 2:
                                break
                    if cnt >= 2:
                        break
            checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': schedeler.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(checkpoint, checkpoint_path)        
        if val_loss<best_loss:
            best_loss=val_loss
            torch.save(model.state_dict(),'best_model.pth')
            print("Model saved!")
            patience_counter=0
        else:
            patience_counter+=1
            if patience_counter>=patience:
                print("Early stopping triggered!")
                break
    # 绘制loss曲线
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    
model=Unet().to(device)
checkpoint_path = 'best_model.pth'   
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded pretrained model from {checkpoint_path}")
else:
    print("No pretrained model found, training from scratch.")


train_model(train_loader, val_loader, model)  





