# ViconToSMPLX
Converts Vicon mocap data to SMPLX

## Quick Start
To run the code on a single take of Vicon data, run the following command:
```bash
python main.py --mocap_file <path_to_psu_tmm>/MOCAP_MRK_1.mat --output_file <path_to_output>/output.pkl
```

For example, to run on subject 1, take 1, run the following command:
```bash
python main.py --mocap_file data_path/Subject_wise/Subject1/MOCAP_MRK_1.mat --output_file output1.pkl
```

## Sources

Regarding the Vicon marker to SMPLX joint mapping, refer to 2.2.3 Preparing the Marker Dataset in Divyesh's thesis. The mapping is stored in `assets/superset.json` and `assets/labeled_TMM100_smplx.json`. Note the mappings are not the exact same and I'm unsure which one is better. I would refer to where Divyesh is creating/using these mappings and see what it's being used for.

The mapping indices come from creating the creating the "labeled_TMM100" dataset in Divyesh's pipeline. Refer to the part on creating the unlabeled and labeled datasets in Divyesh's code ([here](https://github.com/Divyesh-Johri/soma?tab=readme-ov-file#prepare-body-model-and-co))