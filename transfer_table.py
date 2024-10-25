from collections import OrderedDict
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir",type=str,default="log/large")
parser.add_argument("--suffix_list",type=str,nargs="+", default=['','_iter','_sd','_unipc'])
parser.add_argument("--method_list",type=str,nargs="+",default=['calibnet','rggnet','lccnet','lccraft_small','lccraft_large','main_donly','main_ponly','main'])
parser.add_argument("--name_method_list",type=str,nargs="+",default=['CalibNet','RGGNet','LCCNet','LCCRAFT-S','LCCRAFT-L',r'ProjFusion w/o $\bm{F}_P$',r'ProjFusion w/o $\bm{F}_D$',r'ProjFusion'])
parser.add_argument("--name_suffix_list",type=str,nargs="+",default=['',' + Iter',' + NLSD',' + LSD'])
parser.add_argument("--first_suffix",type=str,nargs="+",default=[r'~\cite{CalibNet}',r'~\cite{RGGNet}',r'~\cite{LCCNet}',r'~\cite{LCCRAFT}','','','',''])
parser.add_argument("--key_order",type=str,nargs="+",default=['Rx','Ry','Rz','R','tx','ty','tz','t','3d3c','5d5c'])
parser.add_argument("--save_table",type=str,default="tmp_table.txt")
args = parser.parse_args()
with open(args.save_table,'w') as f:
    for i, (suffix, name_suffix) in enumerate(zip(args.suffix_list, args.name_suffix_list)):
        for j, (method, method_name) in enumerate(zip(args.method_list, args.name_method_list)):
            filename = os.path.join(args.src_dir, method+suffix+'.json')
            assert os.path.isfile(filename), "{} does not exist".format(filename)
            summary = json.load(open(filename,'r'))[-1]
            values = []
            f.write(method_name+name_suffix)
            if i == 0:
                f.write(args.first_suffix[j])
            for key in args.key_order:
                value = summary[key]
                if 't' in key:
                    value *= 100
                    values.append("{:.3f}".format(value))
                elif key == '3d3c' or key == '5d5c':
                    value *= 100
                    values.append("{:.2f}".format(value)+r'\%')
                else:
                    values.append("{:.3f}".format(value))
            f.write(r" &"+r" &".join(values))
            f.write(r" \\")
            f.write('\n')
        if i != len(args.suffix_list)-1:
            f.write(r'\midrule')
            f.write('\n')
        else:
            f.write(r'\bottomrule')
print('Table saved to {}'.format(args.save_table))