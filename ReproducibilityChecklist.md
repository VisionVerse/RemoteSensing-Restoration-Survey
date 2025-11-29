**Table 2.** 
<table>
  <tr>
      <th><b>Model</b></th>
      <th><b>Dataset</b></th>
      <th><b>Model Weights</b></th>
      <th><b>Dependencies</b></th>
      <th><b>Test Code</b></th>
      <th><b>Command Lines</b></th>
      <th><b>Hyperparameter</b></th>
      <th><b>Results</b></td>
  <tr>  
  <tr>
    <td rowspan="2"><b>SpA-GAN (arXiv 2020)</b></td>
    <td>SateHaze1k</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">SpA-GAN SateHaze1k model</a></td>
    <td rowspan="2">python==3.9, pytorch==2.5.0, torchvision==0.20.0, tqdm==4.67.1, numpy==1.23.5</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">SpA-GAN test code</a></td>
    <td><code>python predict.py --config 'config.yml' --test_dir 'data/Test' --out_dir 'result' --pretrained 'pretrained_models/SateHaze1k/gen_model_epoch_200.pth' --cuda</code></td>
    <td>epoch==200, batchsize==1, imgsize==512×512, learning rate==4e-4, seed==0</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">SpA-GAN SateHaze1k result</a></td>
  </tr>
  <tr>
    <td>RRSHID</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">SpA-GAN RRSHID model</a></td>
    <td><code>python predict.py --config 'config.yml' --test_dir 'data/Test' --out_dir 'result' --pretrained 'pretrained_models/RRSHID/gen_model_epoch_200.pth' --cuda</code></td>
    <td>epoch=200, batchsize=1, imgsize=256×256, learning rate=4e-4, seed=0</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">SpA-GAN RRSHID result</a></td>
  </tr>
  <tr>
    <td rowspan="2"><b>DCIL (TGRS 2022)</b></td>
    <td>SateHaze1k</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DCIL SateHaze1k model</a></td>
    <td rowspan="2">python==3.9, pytorch==2.6.0, torchvision==0.21.0, visdom==0.2.4</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DCIL test code</a></td>
    <td><code>python test.py</code></td>
    <td>epoch==200, batchsize==10, imgsize==512×512, learning rate==5e-5</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DCIL SateHaze1k result</a></td>
  </tr>
  <tr>
    <td>RRSHID</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DCIL RRSHID model</a></td>
    <td><code>python test.py</code></td>
    <td>epoch==200, batchsize==10, imgsize==256×256, learning rate==5e-5</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DCIL RRSHID result</a></td>
  </tr>
  <tr>
    <td rowspan="2"><b>DehazeFormer (TIP 2023)</b></td>
    <td>SateHaze1k</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeFormer SateHaze1k model</a></td>
    <td rowspan="2">python==3.7, pytorch==1.10.2, torchvision==0.11.3, tqdm==4.67.1</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeFormer test code</a></td>
    <td><code>python test.py --model dehazeformer-t -- save_dir ./save_models/ --dataset SateHaze1k --exp rshaze</code></td>
    <td>epoch==150, batchsize==32, imgsize==512×512, learning rate==4e-4</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeFormer SateHaze1k result</a></td>
  </tr>
  <tr>
    <td>RRSHID</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeFormer RRSHID model</a></td>
    <td><code>python test.py --model dehazeformer-t -- save_dir ./save_models/ --dataset SateHaze1k --exp rshaze</code></td>
    <td>epoch==150, batchsize==32, imgsize==256×256, learning rate==4e-4</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeFormer RRSHID result</a></td>
  </tr>
  <tr>
    <td rowspan="2"><b>PSMB-Net (TGRS 2023)</b></td>
    <td>SateHaze1k</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">PSMB-Net SateHaze1k model</a></td>
    <td rowspan="2">python==3.7, pytorch==1.8.0, torchvision==0.9.0, timm==1.0.21, tqdm==4.67.1</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">PSMB-Net test code</a></td>
    <td><code>python test.py --test_dir ./datasets_test/ --model_path ./models/SateHaze1k_best.pth --output_dir ./test_results</code></td>
    <td>epoch==200, batchsize==2, imgsize==512×512, learning rate==1e-4</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">PSMB-Net SateHaze1k result</a></td>
  </tr>
  <tr>
    <td>RRSHID</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">PSMB-Net RRSHID model</a></td>
    <td><code>python test.py --test_dir ./datasets_test/ --model_path ./models/RRSHID_best.pth --output_dir ./test_results</code></td>
    <td>epoch==200, batchsize==2, imgsize==256×256, learning rate==1e-4</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">PSMB-Net RRSHID result</a></td>
  </tr>
  <tr>
    <td rowspan="2"><b>Trinity-Net (TGRS 2023)</b></td>
    <td>SateHaze1k</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">Trinity-Net SateHaze1k model</a></td>
    <td rowspan="2">python==3.10, pytorch==2.2.1, torchvision==0.17.1, timm==1.0.20, tqdm==4.67.1, pillow==11.3.0</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">Trinity-Net test code</a></td>
    <td><code>Python Enh_eval.py</code></td>
    <td>epoch==500, batchsize==4, imgsize==512×512, learning rate==5e-4, seed==4</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">Trinity-Net SateHaze1k result</a></td>
  </tr>
  <tr>
    <td>RRSHID</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">Trinity-Net RRSHID model</a></td>
    <td><code>Python Enh_eval.py</code></td>
    <td>epoch==500, batchsize==4, imgsize==256×256, learning rate==5e-4, seed==4</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">Trinity-Net RRSHID result</a></td>
  </tr>
  <tr>
    <td rowspan="2"><b>LFD-Net (JSTARS 2023)</b></td>
    <td>SateHaze1k</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">LFD-Net SateHaze1k model</a></td>
    <td rowspan="2">python==3.9, pytorch==2.6.0, torchvision==0.21.0, pillow==11.3.0, opencv==4.9.0, numpy=1.26.3</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">LFD-Net test code</a></td>
    <td><code>python infer_multi.py -td hazy/ -sd dehaze/</code></td>
    <td>epoch==500, batchsize==8, imgsize==512×512, learning rate==1e-3, seed==1143</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">LFD-Net SateHaze1k result</a></td>
  </tr>
  <tr>
    <td>RRSHID</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">LFD-Net RRSHID model</a></td>
    <td><code>python infer_multi.py -td hazy/ -sd dehaze/</code></td>
    <td>epoch==500, batchsize==8, imgsize==256×256, learning rate==1e-3, seed==1143</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">LFD-Net RRSHID result</a></td>
  </tr>
  <tr>
    <td rowspan="2"><b>ASTA (GRSL 2024)</b></td>
    <td>SateHaze1k</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">ASTA SateHaze1k model</a></td>
    <td rowspan="2">python==3.9, pytorch==2.6.0, torchvision==0.21.0, timm==1.0.22</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">ASTA test code</a></td>
    <td><code>python test.py -- model asta --save_dir ./save_models_SateHaze1k/ --dataset SateHaze1k</code></td>
    <td>epoch==60, batchsize==4, imgsize==512×512, learning rate==2e-4</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">ASTA SateHaze1k result</a></td>
  </tr>
  <tr>
    <td>RRSHID</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">ASTA RRSHID model</a></td>
    <td><code>python test.py -- model asta --save_dir ./save_models_RRSHID/ --dataset RRSHID</code></td>
    <td>epoch==60, batchsize==4, imgsize==256×256, learning rate==2e-4</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">ASTA RRSHID result</a></td>
  </tr>
  <tr>
    <td rowspan="2"><b>DehazeXL (CVPR 2025)</b></td>
    <td>SateHaze1k</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeXL SateHaze1k model</a></td>
    <td rowspan="2">python==3.10, pytorch==2.6.0, torchvision==0.21.0, timm==1.0.20, tqdm==4.67.1, pillow==11.3.0</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeXL test code</a></td>
    <td><code>python test.py --test_img ./datasets/SateHaze1k/test/cloud_L1 --model_path ./checkpoints_SateHaze1k/best.pth --save_dir ./results_directory</code></td>
    <td>epoch==500, batchsize==8, imgsize==512×512, learning rate==2e-4, seed==22</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeXL SateHaze1k result</a></td>
  </tr>
  <tr>
    <td>RRSHID</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeXL RRSHID model</a></td>
    <td><code>python test.py --test_img ./datasets/RRSHID/test/cloud_L1 --model_path ./checkpoints_RRSHID/best.pth --save_dir ./results_directory</code></td>
    <td>epoch==500, batchsize==8, imgsize==256×256, learning rate==2e-4, seed==22</td>
    <td><a href="https://drive.google.com/file/d/1etTpBsWHsrQAGqApj9Qe5X8FStE63iAV/view?usp=sharing">DehazeXL RRSHID result</a></td>
  </tr>
</table>

