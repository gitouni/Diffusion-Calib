import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir",type=str,default="log/multirange")
parser.add_argument("--suffix",type=str,default='_mr5')
parser.add_argument("--name_suffix",type=str,default=" + MR")
parser.add_argument("--method_list",type=str,nargs="+",default=['calibnet','rggnet','lccnet','lccraft_small','lccraft_large','main'])
parser.add_argument("--name_method_list",type=str,nargs="+",default=['CalibNet','RGGNet','LCCNet','LCCRAFT-S','LCCRAFT-L','ProjFusion'])
parser.add_argument("--key_order",type=str,nargs="+",default=['Rx','Ry','Rz','R','tx','ty','tz','t','3d3c','5d5c'])
parser.add_argument("--save_table",type=str,default="tmp_table_mr5.txt")
args = parser.parse_args()
with open(args.save_table,'w') as f:
    suffix = args.suffix
    name_suffix = args.name_suffix
    for j, (method, method_name) in enumerate(zip(args.method_list, args.name_method_list)):
        filename = os.path.join(args.src_dir, method+suffix+'.json')
        assert os.path.isfile(filename), "{} does not exist".format(filename)
        summary = json.load(open(filename,'r'))[-1]
        values = []
        f.write(method_name+name_suffix)
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
    f.write(r'\bottomrule')
print('Table saved to {}'.format(args.save_table))