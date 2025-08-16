This is a wrapper of blood-cell-counter by dnth (https://huggingface.co/spaces/dnth/blood-cell-counter).

How to use: 
1. Clone this git repo on your computer. `git clone https://github.com/dkaxira/cell-counter-wrapper.git`
2. Go into the folder. `cd cell-counter-wrapper`
3. Make the conda environment. `conda env create -f cell-counter-env.yml`
4. Activate the conda environment. `conda activate blood-cell-counter`
5. Make sure all your images are in one folder.
6. Run the script. `python blood_cell_batch.py input/file/path -o my_results.csv -c 0.1 -d 2.0`
