# --*-- coding= utf-8 --*--
# @author: lixinkui
# @time: 20210728

import argparse
import os,errno
import random

import cv2
from core.core import core_handle
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str,
                        help="The output directory", default="demo/out/")
    parser.add_argument("-i", "--input_file", type=str,
                        help="Single file or directory", default="demo/input.png")
    parser.add_argument("-c", "--count", type=int,
                        help="The number of per action to be created.", default=1)
    parser.add_argument("-r", "--random", type=int,
                        help="Whether tps_cv2.", default=0)
    parser.add_argument("-mb", "--motion_blur", type=int,
                        help="Whether motion_blur.", default=1)
    parser.add_argument("-gb", "--gauss_blur", type=int,
                        help="Whether gauss_blur.", default=1)
    parser.add_argument("-sn", "--sp_noise", type=int,
                        help="Whether sp_noise.", default=1)
    parser.add_argument("-gn", "--gauss_noise", type=int,
                        help="Whether gauss_noise.", default=1)
    parser.add_argument("-hz", "--haze", type=int,
                        help="Whether haze.", default=0)
    parser.add_argument("-cl", "--contrast_luminance", type=int,
                        help="Whether contrast_luminance.", default=1)
    parser.add_argument("-rb", "--rot_broa", type=int,
                        help="Whether rot_broa.", default=1)
    parser.add_argument("-at", "--affineTrans", type=int,
                        help="Whether affineTrans.", default=1)
    parser.add_argument("-tc", "--tps_cv2", type=int,
                        help="Whether tps_cv2.", default=1)

    return parser.parse_args()

def prepare(kwargs):
    handle_obj = core_handle()
    file_list = kwargs['count'] * kwargs['input_file']
    for i in tqdm(range(len(file_list))):
        img = cv2.imread(file_list[i])
        name = file_list[i].split('/')[-1].split('.')[0]

        if kwargs['motion_blur']:
            mb = handle_obj.motion_blur(img)
            cv2.imwrite('{}{}_mb_{}.jpg'.format(kwargs["output_dir"],name,
                                                str(i // len(kwargs['input_file']) + 1)), mb)
        if kwargs['gauss_blur']:
            gb = handle_obj.gauss_blur(img)
            cv2.imwrite('{}{}_gb_{}.jpg'.format(kwargs["output_dir"], name,
                                                str(i // len(kwargs['input_file']) + 1)), gb)
        if kwargs['sp_noise']:
            sn = handle_obj.sp_noise(img)
            cv2.imwrite('{}{}_sn_{}.jpg'.format(kwargs["output_dir"], name,
                                                str(i // len(kwargs['input_file']) + 1)), sn)
        if kwargs['gauss_noise']:
            gn = handle_obj.gauss_noise(img)
            cv2.imwrite('{}{}_gn_{}.jpg'.format(kwargs["output_dir"], name,
                                                str(i // len(kwargs['input_file']) + 1)), gn)
        if kwargs['haze']:
            hz = handle_obj.haze(img)
            cv2.imwrite('{}{}_hz_{}.jpg'.format(kwargs["output_dir"], name,
                                                str(i // len(kwargs['input_file']) + 1)), hz)
        if kwargs['contrast_luminance']:
            cl = handle_obj.contrast_luminance(img)
            cv2.imwrite('{}{}_cl_{}.jpg'.format(kwargs["output_dir"], name,
                                                str(i // len(kwargs['input_file']) + 1)), cl)
        if kwargs['rot_broa']:
            rb = handle_obj.rot_broa(img)
            cv2.imwrite('{}{}_rb_{}.jpg'.format(kwargs["output_dir"], name,
                                                str(i // len(kwargs['input_file']) + 1)), rb)
        if kwargs['affineTrans']:
            at = handle_obj.affineTrans(img)
            cv2.imwrite('{}{}_at_{}.jpg'.format(kwargs["output_dir"], name,
                                                str(i // len(kwargs['input_file']) + 1)), at)
        if kwargs['tps_cv2']:
            tc = handle_obj.tps_cv2(img)
            cv2.imwrite('{}{}_tc_{}.jpg'.format(kwargs["output_dir"], name,
                                                str(i // len(kwargs['input_file']) + 1)), tc)



def prepare_random(c,input_,output_):
    handle_obj = core_handle()
    file_list = c * input_
    for i in tqdm(range(len(file_list))):
        img = cv2.imread(file_list[i])
        index = random.randint(1,9)
        name = file_list[i].split('/')[-1].split('.')[0]
        if index == 1: out = handle_obj.motion_blur(img)
        elif index == 2: out = handle_obj.gauss_blur(img)
        elif index == 3: out = handle_obj.sp_noise(img)
        elif index == 4: out = handle_obj.gauss_noise(img)
        elif index == 5: out = handle_obj.haze(img)
        elif index == 6: out = handle_obj.contrast_luminance(img)
        elif index == 7: out = handle_obj.rot_broa(img)
        elif index == 8: out = handle_obj.affineTrans(img)
        elif index == 9: out = handle_obj.tps_cv2(img)
        cv2.imwrite('{}{}_{}.jpg'.format(output_, name,
                                                str(i // len(input_) + 1)), out)


def main():
    args = parse_arguments()
    # output_dir
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # input_file
    input_file = []
    if os.path.isfile(args.input_file):
        input_file.append(args.input_file)
    if os.path.isdir(args.input_file):
        base_dir = args.input_file if args.input_file[-1] == '/' else args.input_file + '/'
        for i in os.listdir(args.input_file):
            input_file.append('{}{}'.format(base_dir,i))

    kw_dic = {}
    kw_dic['output_dir'] = args.output_dir
    kw_dic['input_file'] = input_file
    kw_dic['count'] = args.count
    kw_dic['motion_blur'] = args.motion_blur
    kw_dic['gauss_blur'] = args.gauss_blur
    kw_dic['sp_noise'] = args.sp_noise
    kw_dic['gauss_noise'] = args.gauss_noise
    kw_dic['haze'] = args.haze
    kw_dic['contrast_luminance'] = args.contrast_luminance
    kw_dic['rot_broa'] = args.rot_broa
    kw_dic['affineTrans'] = args.affineTrans
    kw_dic['tps_cv2'] = args.tps_cv2

    if args.random:
        prepare_random(args.count,input_file,args.output_dir)
    else:
        prepare(kw_dic)

if __name__ == "__main__":
    main()