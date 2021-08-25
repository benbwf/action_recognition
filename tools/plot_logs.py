import matplotlib.pyplot as plt
import os
import glob
import re
import argparse
parser = argparse.ArgumentParser(description="Plots train and val losses")
parser.add_argument('input', type=str, help = 'path to log.csv')
args = parser.parse_args()
# input_file = "../log/TSM_ite_RGB_mobilenetv2_shift8_blockres_avg_segment16_e200/log.csv"
# match_text = "Best Prec@1:"
train_loss, train_prec, val_loss, val_prec = list(), list(), list(), list()
with open (args.input, 'r') as file:
#     lines = file.readlines()
    for line in file.readlines():
#         print(line)
        fields = line.split(' ')
        if fields[0] == 'Epoch:':
            tloss = fields[9]
            tloss = float(re.search('\((.*?)\)', tloss).group(1))

            tprec = fields[11]
            tprec = float(re.search('\((.*?)\)', tprec).group(1))
            
        elif fields[0] == 'Testing':
  
            vloss = float(fields[7].strip())
            vprec = float(fields[3])
            train_loss.append(tloss)
            train_prec.append(tprec)
            val_loss.append(vloss)
            val_prec.append(vprec)
            
save_dir = os.path.dirname(args.input)
plt.plot(train_loss, label = 'train loss', marker='', color='blue', linewidth=2)
plt.plot(val_loss,label = 'val loss', marker='', color='red', linewidth=2)
plt.xlabel('epoch')
plt.ylabel('loss')
# show legend
plt.legend()

# show graph
# plt.show()
plt.savefig(os.path.join(save_dir, 'loss.png'))
plt.close()

plt.plot(train_prec, label = 'train prec@1', marker='', color='blue', linewidth=2)
plt.plot(val_prec, label = 'val prec@1', marker='', color='red', linewidth=2)
plt.xlabel('epoch')
plt.ylabel('prec@1')
# show legend
plt.legend()

# show graph
# plt.show()
plt.savefig(os.path.join(save_dir, 'prec1.png'))
plt.close()
        
    