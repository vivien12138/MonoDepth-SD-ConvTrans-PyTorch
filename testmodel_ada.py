from __future__ import print_function
import argparse
import torch.utils.data
from nyuv2_dataset_ada import *
from model_convformer_ada import *
import torch.nn.functional as F
import matplotlib.image as mp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # for multiple GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

DOUBLE_BIAS = 1
WEIGHT_DECAY = 4e-5
distance = 50
normalfactor =  10. # 10. for nyuv2 and  (256.*256.)for kitti
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Single image depth estimation')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='nyuv2', type=str) #'nyuv2''kitti'
    parser.add_argument('--cuda', dest='cuda', help='whether use CUDA', default=True, action='store_true')
    parser.add_argument('--bs', dest='bs', help='batch_size', default=1, type=int)
    parser.add_argument('--num_workers', dest='num_workers', help='num_workers', default=2, type=int)
    parser.add_argument('--result_dir', dest='result_dir', help='output result', default='results', type=str)
    args = parser.parse_args()
    return args


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')

        loss = torch.sqrt( torch.mean( torch.abs(normalfactor*real - normalfactor*fake) ** 2 ) )
        return loss

class RMSEIM(nn.Module):
    def __init__(self):
        super(RMSEIM, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')

        rmseim = torch.abs(normalfactor * real - normalfactor * fake) ** 2
        #print(rmseim.shape)
        return rmseim

class REL(nn.Module):
    def __init__(self):
        super(REL, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        epsilon =torch.from_numpy(1e-4 * np.ones(shape=real.shape)).float().cuda()
        loss = torch.mean(torch.div(torch.abs(real - fake), real + epsilon))
        return loss

class LOG_10(nn.Module):
    def __init__(self):
        super(LOG_10, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        epsilon = torch.from_numpy(1e-4 * np.ones(shape=real.shape)).float().cuda()
        loss = torch.mean(torch.abs(torch.log(torch.abs(normalfactor*real+epsilon)) - torch.log(torch.abs(normalfactor*fake+epsilon))))
        return loss

class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        epsilon = torch.from_numpy(1e-4 * np.ones(shape=real.shape)).float().cuda()
        loss = torch.sqrt(torch.mean(torch.abs(torch.log(torch.abs(normalfactor*real+epsilon)) - torch.log(torch.abs(normalfactor*fake+epsilon))) ** 2))
        return loss

class ACC_thr(nn.Module):
    def __init__(self):
        super(ACC_thr, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        num = np.zeros(3, dtype=np.float32)
        epsilon = 1e-4*torch.ones_like(real, requires_grad=False)
        x = torch.div(real, fake+epsilon)
        y = torch.div(fake, real+epsilon)
        z = torch.cat((x, y), dim=1)
        loss = torch.max(z, dim=1)
        total = float(real.shape[0] * real.shape[1] * real.shape[2] * real.shape[3])
        thr1 = 1.25
        temp = loss.values < thr1
        number = float(temp.nonzero().shape[0])
        num[0] = number/total

        thr2 = 1.25**2
        temp = loss.values < thr2
        number = float(temp.nonzero().shape[0])
        num[1] = number/total

        thr3 = 1.25**3
        temp = loss.values < thr3
        number = float(temp.nonzero().shape[0])
        num[2] = number/total

        return num

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, grad_fake, grad_real):
        prod = torch.matmul(grad_fake[:, :, None, :], grad_real[:, :, :, None]).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt(torch.sum(grad_fake ** 2, dim=-1))
        real_norm = torch.sqrt(torch.sum(grad_real ** 2, dim=-1))

        return 1 - torch.mean(prod / (fake_norm * real_norm))


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        loss = torch.mean(torch.abs(normalfactor*real - normalfactor*fake))
        return loss

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        # fake = torch.where(real < 0, real, fake)  # for KITTI
        loss = torch.sqrt(torch.mean(torch.abs(normalfactor*real - normalfactor*fake) ** 2))
        return loss

class Log_Cosh(nn.Module):
    def __init__(self):
        super(Log_Cosh, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        loss = torch.mean(torch.log(torch.cosh(normalfactor*torch.abs(torch.log(real)) - normalfactor*torch.abs(torch.log(fake)))))
        return loss

class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()

    def forward(self, fake, real):
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        loss = torch.mean(torch.abs(torch.log(real) - torch.log(fake)))
        return loss


class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(self, fake, real):
        threshold = 0.2
        mask = real.values > 0
        if not fake.shape == real.shape:
            _, _, H, W = real.shape
            fake = F.interpolate(fake, size=(H, W), mode='bilinear')
        fake = fake * mask
        diff = torch.abs(real - fake)
        temp = torch.max(diff).detach().cpu().numpy()# [0]
        delta = self.threshold * temp

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 - delta ** 2, 0., -delta ** 2.) + 2.*delta ** 2
        part2 = part2 / (2. * delta)


        loss = part1 + part2
        loss = torch.mean(loss)
        return loss

def adjust_learning_rate(optimizer, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def save_images(img, size, path, gt, step):
    if not img.shape == size:
        H, W = size[2], size[3]
        img = F.interpolate(img, size=(H, W), mode='bilinear')

    im_batch = img.detach().cpu().numpy()

    for i in range(size[0]):

        im = im_batch[i, :, :, :]

        if im.shape[0] == 1:
            if gt:
                im_path = path+'/'+str(step)+'_'+str(i)+'_gt'+'.png'
            else:
                im_path = path+'/'+str(step)+'_'+str(i)+'_pd'+'.png'
            im = im[0, :, :]
            mp.imsave(im_path, im)
        elif im.shape[0] == 3:
            im_path = path + '/' +str(step)+'_'+ str(i) +'_im'+'.png'
            im = im.transpose(1,2,0)
            mp.imsave(im_path, im)

def save_uncertainim(img, size, path, step):
    if not img.shape == size:
        H, W = size[2], size[3]
        img = F.interpolate(img, size=(H, W), mode='bilinear')

    im_batch = img.detach().cpu().numpy()

    for i in range(size[0]):

        im = im_batch[i, :, :, :]
        im_path = path + '/' + str(step) + '_' + str(i) + '_uncer' + '.png'
        im = im[0, :, :]
        mp.imsave(im_path, im)




if __name__ == '__main__':

    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might not want to run with --cuda")

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # dataset
    if args.dataset == 'kitti':
        eval_dataset = KittiDataset(train=False)
        eval_size = len(eval_dataset)
        print(eval_size)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                                                      shuffle=False, num_workers=args.num_workers)

    elif args.dataset == 'nyuv2':
        eval_dataset = NYUv2Dataset(train=False)
        eval_size = len(eval_dataset)
        print(eval_size)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.bs,
                                                      shuffle=False, num_workers=args.num_workers)

    # network initialization
    print('Initializing model...')
    i2d = I2D()
    state = i2d.state_dict()
    checkpoint = torch.load('saved_models/i2d.pth')
    checkpoint = {k.replace('module.',''): v for k, v in checkpoint['model'].items() if k.replace('module.','') in state}
    state.update(checkpoint)
    i2d.load_state_dict(state)
    if 'pooling_mode' in checkpoint.keys():
        POOLING_MODE = checkpoint['pooling_mode']
    print("loaded model")
    #del checkpoint
    torch.cuda.empty_cache()
    i2d = nn.DataParallel(i2d)
    if args.cuda:
        i2d = i2d.cuda()

    print('Done!')


    rmseim = RMSEIM()
    rmse = RMSE()
    rel = REL()
    log_10 = LOG_10()
    rmse_log = RMSE_log()
    thr = ACC_thr()
    depth_criterion = RMSE_log()
    grad_criterion = GradLoss()
    normal_criterion = NormalLoss()
    l2 = L2()
    log_cosh = Log_Cosh()
    berhu = BerHu()

    grad_factor = 1.  # lamda for grad
    normal_factor = 1.

    with torch.no_grad():
        # setting to eval mode
        i2d.eval()

        # img = Variable(torch.FloatTensor(1), volatile=True)
        img = torch.FloatTensor(1)
        img_H = torch.FloatTensor(1)
        # z = Variable(torch.FloatTensor(1), volatile=True)
        z = torch.FloatTensor(1)
        if args.cuda:
            img = img.cuda()
            img_H = img_H.cuda()
            z = z.cuda()

        print('evaluating...')

        rmse_accum = 0.
        rel_accum = 0.
        log_10_accum = 0.
        rmse_log_accum = 0.
        thr_accum = num = np.zeros(3, dtype=np.float32)
        count = 0
        eval_data_iter = iter(eval_dataloader)
        for i, data in enumerate(eval_data_iter):

            img.resize_(data[0].size()).copy_(data[0])
            img_H.resize_(data[1].size()).copy_(data[1])
            z.resize_(data[2].size()).copy_(data[2])

            _, z_fake = i2d(img, img_H)
            # show rmse every image
            u = rmseim(z_fake, z)
            #####################
            rmse_accum += float(img.size(0)) * rmse(z_fake, z) ** 2
            rel_accum += float(img.size(0)) * rel(z_fake, z)
            log_10_accum += float(img.size(0)) * log_10(z_fake, z)
            rmse_log_accum += float(img.size(0)) * rmse_log(z_fake, z) ** 2
            thr_accum[0] += float(img.size(0)) * thr(z_fake, z)[0]
            thr_accum[1] += float(img.size(0)) * thr(z_fake, z)[1]
            thr_accum[2] += float(img.size(0)) * thr(z_fake, z)[2]
            count += float(img.size(0))

            save_images(img, img.shape, args.result_dir, gt=False, step=i)
            save_images(z, z.shape, args.result_dir, gt=True, step=i)
            save_images(z_fake, z.shape, args.result_dir, gt=False, step=i)
            save_uncertainim(u, u.shape, args.result_dir, step=i)

        print(
            "RMSE: %.4f, REL: %.4f, LOG_10: %.4f, RMSE_LOG: %.4f, THR[0]: %.4f, THR[1]: %.4f, THR[2]: %.4f" \
            % (torch.sqrt(rmse_accum / count), (rel_accum / count), (log_10_accum / count),
               torch.sqrt(rmse_log_accum / count), (thr_accum[0] / count), (thr_accum[1] / count),
               (thr_accum[2] / count)))
        with open('val.txt', 'a') as f:
            f.write(
                "RMSE: %.4f, REL: %.4f, LOG_10: %.4f, RMSE_LOG: %.4f, THR[0]: %.4f, THR[1]: %.4f, THR[2]: %.4f\n" \
                % (torch.sqrt(rmse_accum / count), (rel_accum / count), (log_10_accum / count),
                   torch.sqrt(rmse_log_accum / count), (thr_accum[0] / count), (thr_accum[1] / count),
                   (thr_accum[2] / count)))

