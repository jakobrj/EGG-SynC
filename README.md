# EGG-SynC

To run our experiments using conda you have to make sure that conda is installed and do as follows:
<pre>
conda create -n EGGySynC python=3.
conda activate EGGSynC
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install matplotlib
conda install scikitlearn
conda install ninja
</pre>
and run:
<pre>
python run_experiment.py
</pre>