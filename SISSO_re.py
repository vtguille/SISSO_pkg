import os
import shutil
import subprocess
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


"""adding scikit-learn model properties """

EMPTY_SLURM_FN = "/scratch/group/arroyave_lab/\
guillermo.vazquez/SISSO_ver/EMPTY_FILES/runEMPTY.slurm"
EMPTY_SISSO_FN = "/scratch/group/arroyave_lab/\
guillermo.vazquez/SISSO_ver/EMPTY_FILES/SISSO.in"

SLURM_DIC = {
    '--job-name': 'FIRST',
    '--ntasks': '20',
    '--nodes': '1',
    '--ntasks-per-node': '20',
    '--time': '8:00:00',
    '--mail-user': 'guillermo.vazquez@tamu.edu',
    '--mem': '40G',
    
    'spec_modules': ['intel/2022.12'],
    'spec_env_cmd': '',
    'job_commands': ['mpirun -n XX /scratch/group/arroyave_lab/' +
                     'guillermo.vazquez/SISSO_ver/SISSObin/SISSO > log'],
    
    'spec_path': [],
}


SISSO_DIC = {
    'ops': "\'(*)(/)\'",
    'desc_dim': 3,
    # 'funit':"",
    'nf_sis': 50,
    'fcomplexity': 1,
    'method_so': "\'L0\'",
}


class SISSO_obj:
    def __init__(self, directory_filename):
        self.DIR_N = directory_filename
        self.PARENT_DIR = os.getcwd()
        self.STATUS = 'CREATED'
        if os.path.exists(self.DIR_N):
            try:
                os.chdir(self.DIR_N)
                self.load_obj()
                print('object loaded from existing directory')
                os.chdir('..')
            except Exception as e:

                os.chdir('..')
                print('directory exists but NOT able to load from' +
                      ' ls pickle object')
                print(e)

    def set_params(self, slurm_dic=SLURM_DIC, sisso_dic=SISSO_DIC):
        if ~hasattr(self, 'slurm_dic'):
            self.slurm_dic = slurm_dic
        if ~hasattr(self, 'sisso_dic'):
            self.sisso_dic = sisso_dic

    def fit(self, train_df, target_np):
        # train_df may cause troubles with the scikit model

        if self.STATUS != 'CREATED':
            print('SISSO job has already been submitted plase' +
                  ' reset the directory')
            return
        try:
            folder_or_delete(self.DIR_N)
            os.chdir(self.DIR_N)
            if isinstance(train_df, pd.DataFrame):
                self.tdf = train_df.copy()
                
                self.no_feat = self.tdf.columns.shape[0]
                self.org_cols = list(self.tdf.columns)
                self.str_cols = ['feat{:05d}'.format(i) for i in range(
                    self.no_feat)]
                self.tdf.columns = self.str_cols
                self.tdf.insert(0, 'index', self.tdf.index)
                
            else:
                # lists and arrays
                array = np.array(train_df)
                self.no_feat = array.shape[1]
                self.str_cols = ['feat{:05d}'.format(i) for i in range(
                    self.no_feat)]
                self.org_cols = self.str_cols.copy()
                self.tdf = pd.DataFrame(columns=self.str_cols, data=array)
                self.tdf.insert(0, 'index', self.tdf.index)
            if isinstance(target_np, pd.DataFrame):
                self.y = target_np.copy()
            else:
                self.y = pd.Series(target_np)

            self.tdf.insert(1, 'target', self.y.copy())
            self.tdf.to_csv('train.dat', sep=' ', index=False)
            self.create_SLURM(self.slurm_dic)
            self.create_SISSOin(self.sisso_dic)

            self.send_job()
            self.save_obj()
            self.last_dimension = 0
            os.chdir('..')
        except Exception as e:
            print('Exception: \n', e)
            os.chdir('..')

    def create_SLURM(self, dir_options):
        do = dir_options

        with open(EMPTY_SLURM_FN, "r") as in_file:
            buf = in_file.read()

        if len(do['spec_modules']) > 0:
            buf = buf.replace('MODULE LOAD', 'module load ' +
                              ' '.join(do['spec_modules']))

        if len(do['spec_path']) > 0:
            dosp = []
            for i in do['spec_path']:
                dosp.append('\"'+i+':$PATH'+'\"')
                buf = buf.replace('PATH LOAD', '\nexport PATH= ' + '\n\
                    export PATH= '.join(dosp))
        else:
            buf = buf.replace('PATH LOAD', '\n')

        if len(do['spec_env_cmd']) > 0:
            buf = buf.replace('ENV LOAD', do['spec_env_cmd'])
        else:
            buf = buf.replace('ENV LOAD', '')

        # command sen
        if len(do['job_commands']) > 0:
            buf = buf.replace(
                '#---------------------- job ----------------------------#', 
                '#---------------------- job ----------------------------#\n' +
                '\n'.join(do['job_commands']))

            buf = buf.replace('mpirun -n XX', 'mpirun -n '+do['--ntasks'])
        else:
            pass

        buf = buf.split('\n')

        for i, str_i in enumerate(buf):
            for key, value in do.items():
                if '--' in key:
                    if key in str_i:
                        buf[i] = '#SBATCH '+key+'='+value

        buf = [i.split('\n') for i in buf]
        buf = [item for sublist in buf for item in sublist]

        buf = '\n'.join(buf)
        if do['--job-name'] != '':
            with open(do['--job-name']+'.slurm', "w") as out_file:
                out_file.write(buf)
        # print(buf)
        return None

    def create_SISSOin(self, dir_options):
        do = dir_options

        with open(EMPTY_SISSO_FN, "r") as in_file:
            buf = in_file.read()

        buf = buf.split('\n')

        for i, str_i in enumerate(buf):
            
            if str_i.startswith('nsample'):
                buf[i] = 'nsample'+'='+str(len(self.tdf))
            if 'nsf' in str_i:
                buf[i] = 'nsf'+'='+str(self.no_feat)
            if 'funit' in str_i:
                buf[i] = 'funit'+'='+'(1:'+str(self.no_feat)+') '

        for key, value in do.items():
            for i, str_i in enumerate(buf):
            
                if key in str_i:
                    buf[i] = key+'='+str(value)
                    break

        buf = [i.split('\n') for i in buf]
        buf = [item for sublist in buf for item in sublist]

        buf = '\n'.join(buf)
        with open('SISSO.in', "w") as out_file:
            out_file.write(buf)

        # print(buf)
        return None

    # we have enough to start the run here I believe

    def send_job(self):
        send_str = ['sbatch', self.slurm_dic['--job-name']+'.slurm']
        try:

            send_out = subprocess.run(send_str, capture_output=True)
            out_work = send_out.stdout.decode("utf-8").split()

            self.slurm_ID = int(out_work[-1])
            
            print('SLURM submitted with and id of: '+str(self.slurm_ID))
            
            self.STATUS = 'SENT'
        except Exception as e: 
            print('ERROR at sending:\n', e)
            self.STATUS = 'ERROR'
            pass

    def save_obj(self, save_file='obj.pkl'):
        with open(save_file, "wb") as fp:
            d = dict(self.__dict__)
            pickle.dump(d, fp)

    def load_obj(self, save_file='obj.pkl'):
        with open(save_file, "rb") as fp:
            d = pickle.load(fp)
        self.__dict__.update(d)

    def update(self, force_update=False):
        if self.STATUS == 'COMPLETED' and not force_update:
            # print('job already completed and nothing to update')
            return
        try:
            send_str = ['sacct', '-j', str(self.slurm_ID), '-o', 'state']
            send_out = subprocess.run(send_str, capture_output=True)
            out_work = send_out.stdout.decode("utf-8").split()
            self.STATUS = out_work[2]
        except Exception as e:
            print('SLURM job STATUS is not ready or not working: ')
            print(e)
        
        if os.path.exists(self.DIR_N + '/SISSO.out'):
            with open(self.DIR_N + '/SISSO.out', "r") as in_file:
                self.SISSO_out = in_file.read()
        else:
            self.last_dimension = 0
            print('no SISSO.out file created yet')
            return 

        all_lines = self.SISSO_out.split('\n')
        self.raw_desc = []
        self.raw_model = []
        
        for d in range(1, self.sisso_dic['desc_dim']+1):
            
            pattern = str(d)+"D descriptor"
            
            for current_line_no, current_line in enumerate(all_lines):
                if pattern in current_line:
                    for k in range(d):
                        self.raw_desc.append(
                            all_lines[current_line_no+1+k]
                            .split('=')[1].split(' ')[1])
                    break

            pattern = str(d)+"D model"
            
            for current_line_no, current_line in enumerate(all_lines):
                if pattern in current_line:
                    for k in range(3):
                        self.raw_model.append(
                            [float(n) for n in all_lines[
                             current_line_no + 1 + k]
                             .split(':')[1].split()])
                    break
            self.last_dimension = int(len(self.raw_model)/3)
            if self.last_dimension != d:
                break
        
        print('Reading SISSO.out till dimension ' +
              str(self.last_dimension) + ' out of ' +
              str(self.sisso_dic['desc_dim']))
        
        used_feat = []

        v_lines = []
        for i, _ in enumerate(self.raw_desc):
            v_lines.append(self.raw_desc[i])
            v_lines[i] = v_lines[i]
            # self.v_wo_df.append(v_lines[i][:])
            v_lines[i] = v_lines[i].replace('^', '**')
            v_lines[i] = v_lines[i].replace('exp', 'np.exp')
            v_lines[i] = v_lines[i].replace('sqrt', 'np.sqrt')
            v_lines[i] = v_lines[i].replace('log', 'np.log')
            v_lines[i] = v_lines[i].replace('abs', 'np.abs')
            v_lines[i] = v_lines[i].replace('cbrt', 'np.cbrt')

            for nf, f in enumerate(self.str_cols):
                if f in v_lines[i]:
                    if f not in used_feat:
                        used_feat.append(f)

                    v_lines[i] = v_lines[i].replace(
                        f, 'df[\''+f+'\'].values')

        self.v_lines = v_lines
        self.save_obj(self.DIR_N+'/obj.pkl')

    def eval_X(self, X, DIM):
        self.update()
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            X = np.array(X)
            df = pd.DataFrame(data=X)
        no_feat = df.columns.shape[0]

        if no_feat != self.no_feat:
            print('ERROR: Dataframes don\'t match')
            return
        df.columns = self.str_cols
        if DIM > self.last_dimension:
            print('ERROR: dimension not available so far or not available')
            return
        elif DIM == 0:
            print('ERROR: dimension is ZERO')
            return

        triangular_number = (DIM - 1) * (DIM) // 2
        suma = self.raw_model[(DIM - 1) * 3 + 1]
        for i, c in enumerate(self.raw_model[(DIM - 1) * 3]):
            suma = suma+c*eval(self.v_lines[triangular_number + i])
        return suma

    def predict(self, X):
        return self.eval_X(X, self.sisso_dic['desc_dim'])

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        return {'RMSE': np.sqrt(mse(y_pred, y)),
                'MAE': np.max(np.abs(y_pred-y)),
                'r2': r2_score(y_pred, y)}

    def print_formula_latex(self, DIM=3):
        triangular_number = (DIM - 1) * (DIM) // 2
        a0 = self.raw_model[(DIM - 1) * 3 + 1]
        a = []
        feats = []
        for i, c in enumerate(self.raw_model[(DIM - 1) * 3]):
            feats.append(self.raw_desc[triangular_number + i])
            a.append(c)
        print(a0, a, feats)
        
        for i in range(DIM):
            
            for f, l in zip(self.str_cols, self.org_cols):
                feats[i] = feats[i].replace(f, l)
            feats[i] = feats[i].replace('log', '\\ln')
            feats[i] = feats[i].replace('_', '\\_')
            feats[i] = feats[i].replace('/', ' / ')
            feats[i] = feats[i].replace('*', ' ')
            
        ws = "{:.2E}".format(a0[0]).replace('E', '\\times 10 ^')
        dummy = ws.split('^')[1]
        intdummy = int(dummy)
        ws = ws.replace(dummy, '{'+str(intdummy)+'}')
        ws = ws.replace('10 ^{0}', '')

        final_str = '$\\begin{array}{c}\n'+ws[:]
        for i in range(len(feats)):
            
            ws = "{:.2E}".format(a[i]).replace('E', '\\times 10 ^')
            dummy = ws.split('^')[1]
            intdummy = int(dummy)
            ws = ws.replace(dummy, '{'+str(intdummy)+'}')
            ws = ws.replace('10 ^{0}', '')
            
            if a[i] < 0:
                final_str = final_str+ws+feats[i]+'\\\\\n'
            else:
                final_str = final_str+"+"+ws+feats[i]+'\\\\\n'
        final_str = final_str+'\\end{array}$'
        print(final_str)


def folder_or_delete(dir_name, delete=True):
    '''checks if there's a floder with that name, if not it creates it,
    if yes it deletes it and it creates it again'''

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        if delete:
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)
        else:
            pass


def get_lims(xs, ys, panf=0.05):
    
    h_ = np.append(xs, ys)
    mi, ma = np.min(h_), np.max(h_)
    pan = panf*(ma-mi)
    return mi-pan, ma+pan


def parity_plots(gtcl_list, s=5, color='blue', alpha=1):
    no_values = len(gtcl_list)
    fig, axs = plt.subplots(1, no_values, figsize=(5*no_values, 5))  
    # Set the figsize parameter to control the figure size
    if no_values == 1:
        ii = gtcl_list[0][0]
        jj = gtcl_list[0][1]
        axs.scatter(ii, jj, s=s, color=color, alpha=alpha)

        mi, ma = get_lims(ii, jj)
        axs.set_xlim(mi, ma)
        axs.set_ylim(mi, ma)
        axs.grid(True)
        axs.annotate(r'$r^2=$'+'{:.4f}'.format(
            r2_score(ii, jj)), xy=(0.1, 0.9), xycoords='axes fraction')
        axs.annotate(r'$RMSE=$'+'{:.4f}'.format(
            np.sqrt(mse(ii, jj))), xy=(0.1, 0.8), xycoords='axes fraction')
        axs.annotate(r'$MAE=$'+'{:.4f}'.format(
            mae(ii, jj)), xy=(0.1, 0.7), xycoords='axes fraction')
    else:
        for i in range(no_values):

            ii = gtcl_list[i][0]
            jj = gtcl_list[i][1]
            axs[i].scatter(ii, jj, s=s, color=color, alpha=alpha)

            mi, ma = get_lims(ii, jj)
            axs[i].set_xlim(mi, ma)
            axs[i].set_ylim(mi, ma)
            axs[i].grid(True)
            axs[i].annotate(r'$r^2=$'+'{:.4f}'.format(
                r2_score(ii, jj)), xy=(0.1, 0.9), xycoords='axes fraction')
            axs[i].annotate(r'$RMSE=$'+'{:.4f}'.format(
                np.sqrt(mse(ii, jj))), xy=(0.1, 0.8), xycoords='axes fraction')
            axs[i].annotate(r'$MAE=$'+'{:.4f}'.format(
                mae(ii, jj)), xy=(0.1, 0.7), xycoords='axes fraction')
