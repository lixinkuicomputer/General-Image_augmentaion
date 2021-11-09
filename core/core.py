# --*-- coding= utf-8 --*--
# @author: lixinkui
# @time: 20210728

import cv2
import numpy as np
import random
from math import fabs, sin, cos, radians
from scipy.stats import mode

class core_handle():
    def __init__(self):
        pass

    # 运动模糊
    def motion_blur(self, image, degree_start=10, degree_stop=25, angle=45):
        '''
        degree: 旋转中心
        angle: 旋转角度
        '''
        degree = random.randint(degree_start,degree_stop)
        image = np.array(image)
        # 生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred

    # 高斯模糊
    def gauss_blur(self, image, ksize_start=3, ksize_stop=20, sigmaX=10, sigmaY=10):
        '''
        ksize: 高斯内核大小
        sigmaX: X方向高斯核标准偏差
        sigmaY: Y方向高斯核标准偏差
        '''
        ksize_single = random.sample(range(ksize_start,ksize_stop,2),1)[0]
        ksize = (ksize_single,ksize_single)
        img = cv2.GaussianBlur(image, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
        return img

    # 椒盐噪声
    def sp_noise(self, image, prob_start=0.02, prob_stop=0.1):
        '''
        prob:噪声比例
        '''
        prob = random.uniform(prob_start,prob_stop)
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    # 高斯噪声
    def gauss_noise(self, image, mean=0, var_start=0.001, var_stop=0.01):
        '''
        mean: 均值
        var: 方差
        '''
        var = random.uniform(var_start,var_stop)
        image = np.array(image/255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out*255)
        return out

    # 雾霾
    def haze(self, image, t_start=0.1, t_stop=0.5, A=1.0):
        '''
        t: 透视率 0~1
        A: 大气光照
        '''
        t = random.randint(10*t_start,10*t_stop) / 10
        out = image * t + A * 255 * (1 - t)
        return out

    # 对比度和亮度
    def contrast_luminance(self, image, cont=0.6, bright=0.6):
        '''
        cont : 对比度，调节对比度应该与亮度同时调节
        bright : 亮度
        '''
        cont_ = random.randint(1,10*cont) / 10
        bright_ = random.randint(1,10*bright) / 10
        out = np.uint8(np.clip((cont_ * image + bright_), 0, 255))
        return out

    # 旋转
    def rot_broa(self, img, degreestart=45, degreestop=330, filled_color=1):
        '''
        degreestart: 旋转区间开始角度
        degreestop: 旋转区间结束角度
        filled_color: 默认填充背景色
        '''
        degree = random.randint(degreestart,degreestop)
        # 获取旋转后4角的填充色
        if filled_color == 1:
            filled_color = mode([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]).mode[0]
        if np.array(filled_color).shape[0] == 2:
            if isinstance(filled_color, int):
                filled_color = (filled_color, filled_color, filled_color)
        else:
            filled_color = tuple([int(i) for i in filled_color])
        height, width = img.shape[:2]
        # 旋转后的尺寸
        height_new = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        width_new = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        mat_rotation[0, 2] += (width_new - width) / 2
        mat_rotation[1, 2] += (height_new - height) / 2
        img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new), borderValue=filled_color)
        # 填充四个角
        mask = np.zeros((height_new + 2, width_new + 2), np.uint8)
        mask[:] = 0
        seed_points = [(0, 0), (0, height_new - 1), (width_new - 1, 0), (width_new - 1, height_new - 1)]
        for i in seed_points:
            cv2.floodFill(img_rotated, mask, i, filled_color)
        return img_rotated

    # 扭曲 (仿射)
    def affineTrans(self, img, p=4):
        '''
        p: 扭曲程度区间
        '''
        rows, cols = img.shape[:2]
        list1 = [20,20,30,20,20,0]
        list2 = [i + random.sample(range(p),1)[0] for i in list1]
        list1_ = [list1[i*2:(i+1)*2] for i in range(int(len(list1)/2))]
        list2_ = [list2[i*2:(i+1)*2] for i in range(int(len(list2)/2))]
        pts1 = np.float32(list1_)
        pts2 = np.float32(list2_)
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    # 扭曲 (tps变换)
    def tps_cv2(self, img, N=50, PADDINGSIZ=80):
        '''
        N: 基准点个数
        PADDINGSIZ: 变换程度
        '''
        # N对基准控制点
        points = []
        dx = int(img.shape[0] / (N - 1))
        for i in range(2 * N):
            points.append((dx * i, 20))
            points.append((dx * i, img.shape[1] - 20))
        img = cv2.copyMakeBorder(img, 4, 4, 0, 0, cv2.BORDER_REPLICATE)
        tps = cv2.createThinPlateSplineShapeTransformer()
        sourceshape = np.array(points, np.int32)
        sourceshape = sourceshape.reshape(1, -1, 2)
        matches = []
        for i in range(1, N + 1):
            matches.append(cv2.DMatch(i, i, 0))
        # 随机变动
        newpoints = []
        for i in range(N):
            nx = points[i][0] + random.randint(0, PADDINGSIZ) - PADDINGSIZ / 2
            ny = points[i][1] + random.randint(0, PADDINGSIZ) - PADDINGSIZ / 2
            newpoints.append((nx, ny))
        targetshape = np.array(newpoints, np.int32)
        targetshape = targetshape.reshape(1, -1, 2)
        tps.estimateTransformation(sourceshape, targetshape, matches)
        img = tps.warpImage(img)
        return img



if __name__ == "__main__":
    handle_obj = core_handle()
    img = cv2.imread('../demo/input.png')
    out = handle_obj.contrast_luminance(img)
    cv2.imwrite('../demo/out.jpg', out)