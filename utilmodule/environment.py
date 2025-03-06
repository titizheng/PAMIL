import numpy as np
import torch 


def expand_data(update_coords, update_datas,action_size = None, total_steps=None):
    """
    Expand the data to reach a specific number of collection steps.

    Parameters:
        data (np.array): The original data.
        step (int): The collection step size.
        total_steps (int): The total number of collection steps, default is 10.
    """
    
    total_length = action_size * total_steps
    B, current_length, C = update_coords.shape
    
    if current_length >= total_length:
        StopT = int(update_coords.shape[1] / action_size)  #+ 1
        remaining_length = current_length - action_size * StopT

        if remaining_length > 0:
            required_length = action_size - remaining_length
            segment_length = 1  
            num_segments = current_length
            random_indices = np.random.choice(num_segments, size=required_length, replace=False)
            random_coords_segments = [update_coords[:, i:i+segment_length, :] for i in random_indices]
            random_data_segments = [update_datas[:, i:i+segment_length, :] for i in random_indices]
            
            random_coords = torch.cat(random_coords_segments, dim=1)
            random_datas = torch.cat(random_data_segments, dim=1)

            update_coords = torch.cat([update_coords, random_coords], dim=1)
            update_datas = torch.cat([update_datas, random_datas], dim=1)

        return update_coords, update_datas, StopT
    else:
    
        repeat_times = total_length // current_length
        
        update_coords = update_coords.repeat(1, repeat_times, 1) 
        update_datas = update_datas.repeat(1, repeat_times, 1) 

        B, current_length, C = update_coords.shape
        remaining_length = total_length - current_length

        segment_length = 1  
        num_segments = current_length
        random_indices = np.random.choice(num_segments, size=remaining_length, replace=False)
        
        random_coords_segments = [update_coords[:, i:i+segment_length, :] for i in random_indices]
        random_data_segments = [update_datas[:, i:i+segment_length, :] for i in random_indices]
        
        random_coords = torch.cat(random_coords_segments, dim=1)
        random_data = torch.cat(random_data_segments, dim=1)

        update_coords = torch.cat([update_coords, random_coords], dim=1)
        update_datas = torch.cat([update_datas, random_data], dim=1)

        StopT = int(update_coords.shape[1] / action_size) 
        
        return update_coords, update_datas, StopT