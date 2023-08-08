import subprocess
import os
import psutil
if __name__ == "__main__":
   logdirs = ['/home/kgalassi/gym-cloth-logs/cascade/dagger-tier-3-pol-pull-cascade-network',
            '/home/kgalassi/gym-cloth-logs/image/dagger-tier-3-pol-pull-image-network',
            '/home/kgalassi/gym-cloth-logs/train/dagger-joint-tier-3-pol-pull-network',
            '/home/kgalassi/gym-cloth-logs/image/dagger-image-network2',
            ]
   ports = [6020, 6050, 6090, 6110]

   log_path = '/home/kgalassi/code/cloth/cloth_training/cloth_training/model/logs'

   def terminate_process_on_port(port):
      for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
         if proc.info['name'] == 'tensorboard' and f'--port={port}' in proc.info['cmdline']:
            proc.terminate()
            print(f"TensorBoard process on port {port} terminated.")



   def launch_tensorboard(logdir, port):
      #path = os.path.join(log_path, logdir)
      terminate_process_on_port(port)
      command = f"tensorboard --logdir={logdir} --bind_all --port={port} --reload_multifile=true"
      
      print('Dir {} : {}'.format(logdir, command))
      subprocess.Popen(command, shell=True)



   for dir, port in zip(logdirs, ports):
      launch_tensorboard(dir, port)


   print('done') 
   print('link are')

   for dir, port in zip(logdirs, ports):
      print(f'http://gargas.ad.europe.naverlabs.com:{port}/#scalars')





#tensorboard --logdir=/home/kgalassi/gym-cloth-logs/train/dagger-joint-tier-3-pol-pull-network/results --bind_all --port=6090 --reload_multifile=true
