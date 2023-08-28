import os
import shutil
import subprocess
import pickle
import numpy as np

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
    'job_commands': ["mpirun -n XX /scratch/group/arroyave_lab/\
    guillermo.vazquez/SISSO_ver/SISSObin/SISSO > log"],
    
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
        self.slurm_dic = slurm_dic
        self.sisso_dic = sisso_dic

    def train(self, train_df, target_np):
        # train_df may cause troubles with the scikit model

        if self.STATUS != 'CREATED':
            print('SISSO job has already been submitted plase' +
                  ' reset the directory')
            return
        try:
            folder_or_delete(self.DIR_N)
            os.chdir(self.DIR_N)
            
            self.tdf = train_df.copy()
            self.y = target_np
            self.no_feat = self.tdf.columns.shape[0]
            self.org_cols = list(self.tdf.columns)
            self.str_cols = ['feat{:05d}'.format(i) for i in range(
                self.no_feat)]
            self.tdf.columns = self.str_cols
            self.tdf.insert(0, 'index', self.y.index)
            self.tdf.insert(1, 'target', self.y.copy())
            
            self.tdf.to_csv('train.dat', sep=' ', index=False)
            self.create_SLURM(self.slurm_dic)
            self.create_SISSOin(self.sisso_dic)

            self.send_job()
            self.save_obj()
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

    def save_obj(self):
        with open('obj.pkl', "wb") as fp:
            d = dict(self.__dict__)
            pickle.dump(d, fp)

    def load_obj(self):
        with open('obj.pkl', "rb") as fp:
            d = pickle.load(fp)
        self.__dict__.update(d)

    def update(self):
        if self.STATUS == 'COMPLETED':
            print('job already completed and nothing to update')
            return
        send_str = ['sacct', '-j', str(self.slurm_ID), '-o', 'state']
        send_out = subprocess.run(send_str, capture_output=True)
        out_work = send_out.stdout.decode("utf-8").split()
        self.STATUS = out_work[2]
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
        self.last_dimension = 0
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

            for nf, f in enumerate(self.str_cols):
                if f in v_lines[i]:
                    if f not in used_feat:
                        used_feat.append(f)

                    v_lines[i] = v_lines[i].replace(
                        f, 'df[\''+f+'\'].values')

        self.v_lines = v_lines

    def eval_df(self, df, DIM):
        
        no_feat = df.columns.shape[0]

        if no_feat != self.no_feat:
            print('ERROR: Dataframes don\'t match')
            return
        df.columns = self.str_cols
        if DIM < 1 or DIM > self.last_dimension:
            print('ERROR: dimension not available so far or not available')
            return

        triangular_number = (DIM - 1) * (DIM) // 2
        print(triangular_number)
        suma = self.raw_model[(DIM - 1) * 3 + 1]
        for i, c in enumerate(self.raw_model[(DIM - 1) * 3]):
            suma = suma+c*eval(self.v_lines[triangular_number + i])
        return suma


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
