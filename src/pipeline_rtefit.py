#This python script was originally created by William Boyes (GA) and
# then edited by Javier Hernandez-Nicolau (SDSC) to run on different HPC systems
# and check Toksearch performance.

import os
from profile import Profile

from toksearch import MdsSignal, Pipeline 
import pdb

import numpy as np
import xarray as xr

import gc
import pickle
import collections
import pprint
# import h5py
import time

import argparse

efit_tree = 'efitrt1'
# define working dir relative to present file
script_path = os.path.abspath(__file__)
base_dir= os.path.dirname(script_path)
os.chdir(base_dir)

# Update directory for individual shot storage
save_dir = f'{base_dir}/shot_results_{efit_tree}'
os.makedirs(save_dir, exist_ok=True)

start_time = time.time()

np.set_printoptions(threshold=3, precision=1)

r_geo = 1.68
mu_0 = 4e-7 * np.pi

def create_efit_pipeline(shots):
	pipe = Pipeline(shots)

	sigs_dict = {}
	loc="remote://atlas.gat.com"
     #comment out location for testing
	sigs_dict["q"] = MdsSignal(r'\QPSI', efit_tree, dims=("psi","times"), data_order=("times","psi"))#, location="remote://atlas.gat.com")
	sigs_dict["p"] = MdsSignal(r'\PRES', efit_tree, dims=("psi","times"), data_order=("times","psi"))#, location="remote://atlas.gat.com")
	sigs_dict["RBBBS"] = MdsSignal(r'\TOP:RESULTS:GEQDSK:RBBBS', efit_tree, dims=("nbbbs", "times"))#, data_order=("times", "nbbbs"), location="remote://atlas.gat.com")
	sigs_dict["ZBBBS"] = MdsSignal(r'\TOP:RESULTS:GEQDSK:ZBBBS', efit_tree, dims=("nbbbs", "times"))#, data_order=("times", "nbbbs"), location="remote://atlas.gat.com")
	sigs_dict["aminor"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:AMINOR', efit_tree)#,location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["bcentr"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:BCENTR', efit_tree)#,location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["kappa"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:KAPPA', efit_tree)#,location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["q0"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:Q0', efit_tree)#, location="remote://atlas.gat.com") 
	sigs_dict["q95"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:q95', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["li"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:li', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["betap"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:betap', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["betat"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:betat', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["tritop"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:tritop', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["tribot"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:tribot', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["ip"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:ipmeas', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["r0"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:R0', efit_tree)#, location="remote://atlas.gat.com") 
	# sigs_dict["density"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:DENSITY', efit_tree, location="remote://atlas.gat.com") 
	sigs_dict["betan"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:betan', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["zcur"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:zcur', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order

	# Use EFIT01, let times get aligned
	sigs_dict["density"] = MdsSignal(r'\density', 'EFIT01')#, location="remote://atlas.gat.com")

	# Try to fetch EFS versions of zcur -> this does not work
	# sigs_dict["zcur"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:efszcur', efit_tree, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order

	# These have no EFS equivalent, will be filled by 0
	sigs_dict["s1"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:s1', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["s2"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:s2', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	sigs_dict["s3"] = MdsSignal(r'\TOP:RESULTS:AEQDSK:s3', efit_tree)#, location="remote://atlas.gat.com") # over time, doesn't need dims or data_order
	


	print(f'{len(sigs_dict)} signals in dataset')
	pipe.fetch_dataset("ds", sigs_dict) 

	# Change time basis to ms for density only 
	@pipe.map
	def convert_efit01_to_ms(record):
		ds = record['ds']
		if ds.density.times.values[0] < 20:
			new_times = ds.density.times * 1000.0
			ds['density'] = ds.density.assign_coords(times=new_times)
		record['ds'] = ds
		return record

	# align now that everything is the same basis 
	pipe.align("ds", "q", method = 'linear') #Aligns all signals to the same times and interpolates.  

	@pipe.map
	def calculate_geometric_and_profiles(record):
		if 'ds' not in record or isinstance(record['ds'], Exception):
			return record

		ds = record['ds']
		# Explicitly save the time base as a numpy array
		record['times'] = ds.times.values

		# Use .values on everything to strip xarray metadata

		r_boundary = ds['RBBBS']
		z_boundary = ds['ZBBBS']
		record['rmax'] = r_boundary.where(r_boundary > 0).max(dim='nbbbs').values
		record['rmin'] = r_boundary.where(r_boundary > 0).min(dim='nbbbs').values

		# record['rmax'] = ds['RBBBS'].max(dim='nbbbs').values
		# record['rmin'] = ds['RBBBS'].min(dim='nbbbs').values
		record['zmax'] = ds['ZBBBS'].max(dim='nbbbs').values
		record['zmin'] = ds['ZBBBS'].min(dim='nbbbs').values

		record['q0'] = np.abs(ds['q'].interp(psi=0.0)).values
		record['q33'] = np.abs(ds['q'].interp(psi=0.33)).values
		record['q67'] = np.abs(ds['q'].interp(psi=0.67)).values

		# Correctly call np.abs() and extract values
		ip_val = np.abs(ds['ip']).values
        
		# produce nan in a number of denominators to remove later, to avoid errors
		bcentr_val = np.abs(ds['bcentr']).values.copy()
		bcentr_val[bcentr_val == 0] = np.nan
            
		aminor_val = ds['aminor'].values.copy()
		aminor_val[aminor_val == 0] = np.nan

		record['aspr'] = r_geo / aminor_val
		record['kappa'] = ds['kappa'].values
		record['iphat'] = (ip_val * mu_0) / (bcentr_val * aminor_val)
		record['q95'] = np.abs(ds['q95']).values
		
		li_val = ds['li'].values.copy()
		li_val[li_val == 0] = np.nan
		record['elli'] = li_val
		
		record['betap'] = ds['betap'].values
		record['betat'] = ds['betat'].values

		b0_sq = bcentr_val**2
		pres0_norm = (2 * mu_0 * ds['p'].interp(psi=0.0).values) / b0_sq        
		pres50_norm = (2 * mu_0 * ds['p'].interp(psi=0.25).values) / b0_sq

		record['dq67'] = np.abs(ds['q'].differentiate('psi').interp(psi=0.67)).values

		# Calculate derivative and ensure it is a numpy array
		dp_dpsi = ds['p'].differentiate('psi').interp(psi=0.25).values
		dp_drho = dp_dpsi * (2 * 0.5)
		dpres50_norm = dp_drho * (2 * mu_0) / b0_sq

		record['tritop'] = ds['tritop'].values
		record['tribot'] = ds['tribot'].values 

		record['nhat'] = (ds["density"].values/1e14) / ( (ip_val/1e6) / (np.pi * (aminor_val**2)) )
		rcur = ds['r0'].values / r_geo
		record['rcur'] = rcur
		record['zcur'] = ds['zcur'].values

		s1, s2 = ds['s1'].values, ds['s2'].values
		record['glitch'] = (s1/4 + (s2/4)*(1 + rcur)) - li_val/2 - ds["betap"].values
		record['shaf1'] = s1
		record['shaf2'] = s2
		record['shaf3'] = ds['s3'].values

		record['pres0'] = pres0_norm * 100
		record['pres50'] = pres50_norm * 100
		record['dpres50'] = dpres50_norm * 100

		# adding features I think should matter
		record['betan'] = ds['betan'].values
		record['nwl_prox'] = ds['betan'].values/(4*li_val)


		if 'ds' in record:
			del record['ds']
		return record

	return pipe

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Run EFIT pipeline with chunking.")
	parser.add_argument('--chunk_size', type=int, default=2, help='Number of shots per chunk (default: 2)')
	parser.add_argument('--num_workers', type=int, default=20, help='Number of workers for multiprocessing (default: 20)')
	args = parser.parse_args()

	file_path = '../CAKE_MASTER_SHOTLIST.pkl'
	with open(file_path, 'rb') as file:
		full_shotlist = pickle.load(file)

	#use smaller set for testing
	shotlist = full_shotlist[:500]
	
    	# Get Slurm Task info
	task_id = int(os.environ.get('SLURM_PROCID', 0))
	num_tasks = int(os.environ.get('SLURM_NTASKS', 1))

	all_shot_chunks = np.array_split(shotlist, num_tasks)
	my_shots = all_shot_chunks[task_id]

	print(f"Task {task_id}/{num_tasks} starting. Shots to process: {len(my_shots)}")

	chunk_size = args.chunk_size
	
	for i in range(0, len(my_shots), chunk_size):
		stop_idx = min(i + chunk_size, len(my_shots))
		current_chunk = my_shots[i:stop_idx]
		
		print(f"--- Task {task_id}: Processing chunk {i//chunk_size} ({len(current_chunk)} shots) ---", flush=True)

		# 1. Create and execute pipeline
		pipe = create_efit_pipeline(current_chunk)
		chunk_results = pipe.compute_multiprocessing(num_workers=args.num_workers)

		# 2. Iterate through the results and save SHOT-BY-SHOT
		#for record in chunk_results:
			# Safely get the shot number from the record
		#	shot_num = record.get('shot', None)
		#	if shot_num is None:
				# Fallback if 'shot' isn't a top-level key (sometimes it's in ds)
		#		try:
		#			shot_num = int(record['ds'].shot.values)
		#		except:
		#			continue

		#	shot_file = os.path.join(save_dir, f'shot_{shot_num}.pkl')

			# Save individual shot record
		#	with open(shot_file, 'wb') as f:
		#		pickle.dump(record, f)

		print(f"Task {task_id}: Finished and saved chunk {i//chunk_size}", flush=True)

		# 3. Clean up memory for the next chunk
		del pipe
		del chunk_results
		gc.collect()

	end_time = time.time()
	print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")
