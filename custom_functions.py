
import os
import json
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfea
import pyproj
import imageio.v2 as imageio
from io import BytesIO
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import zarr # Require dependencies
import pypalettes 
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import math

####################################################################################################################################

####################################################################################################################################

def create_waterlevel_gif(ds,variable, start_time, end_time, station_index, gif_filename,cmap, vmin, vmax):

    station_x = ds['station_x_coordinate'].values
    station_y = ds['station_y_coordinate'].values
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    station_x_3857, station_y_3857 = transformer.transform(station_x, station_y)

    highlight_x = station_x_3857[station_index]
    highlight_y = station_y_3857[station_index]

    bbox = [-20000000, 20000000, -20000000, 20000000]
    projection = ccrs.epsg(3857)
    
    frames = []

    times = ds['time'].sel(time=slice(start_time, end_time))

    for target_time in times:
        waterlevel_at_time = ds[variable].sel(time=target_time).values

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent(bbox, projection)

        land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='dimgray', facecolor='white', zorder=2)
        ax.add_feature(land)
        ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='dimgray', zorder=3)

        title = f'{variable} at {np.datetime_as_string(target_time.values, unit="s")}'
        ax.set_title(title, loc='left', fontsize=9)

        scatter = ax.scatter(station_x_3857, station_y_3857, c=waterlevel_at_time, label='Water Level', cmap=cmap, marker='o', s=20, vmin=vmin, vmax=vmax, zorder=1)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(variable)

        ax.scatter(highlight_x, highlight_y, facecolor='none', edgecolor='red', s=100, marker='s', label=f'Station {station_index}', zorder=4)

        plt.xlabel('Longitude (degrees_east)')
        plt.ylabel('Latitude (degrees_north)')
        plt.legend()
        plt.grid(True)

        # Salva il frame in memoria
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    # Crea la GIF
    imageio.mimsave(gif_filename, frames, duration=0.5)


####################################################################################################################################

####################################################################################################################################

def get_min_max(ds,ds_type,output_path, variable):

    if ds_type=="tide-gauges": 
        # Save locally netCDF file
        filename = 'min_max_tide-gauge.json'
        # Check file
        file_exists = os.path.isfile(output_path+filename)
        if file_exists==False:
             print(filename+" does not exist")
        if file_exists==True:
            # Read
            with open(output_path+filename, 'r') as file:
                coord_min_max = json.load(file)

            vmin = float(coord_min_max["min"])
            vmax = float(coord_min_max["max"])

    if ds_type=="gridded":
        # Save locally netCDF file
        filename = 'min_max_gridded.json'
        # Check file
        file_exists = os.path.isfile(output_path+filename)
        if file_exists==False:
             print(filename+" does not exist")
        if file_exists==True:
            # Read
            with open(output_path+filename, 'r') as file:
                coord_min_max = json.load(file)

            vmin = float(coord_min_max["min"])
            vmax = float(coord_min_max["max"])

    if ds_type=="ERA5-forcing": 
        vmin = np.ones([3])
        vmin = np.ones([3])
        # Save locally netCDF file
        filename = 'min_max_ERA5_forcing.json'
        # Check file
        file_exists = os.path.isfile(output_path+filename)
        if file_exists==False:
            print(filename+" does not exist")
        if file_exists==True:
            # Read
            with open(output_path+filename, 'r') as file:
                coord_min_max = json.load(file)

            vmin = [float(coord_min_max["min_wind"]),float(coord_min_max["min_tmp"]),float(coord_min_max["min_msl"])]
            vmax = [float(coord_min_max["max_wind"]),float(coord_min_max["max_tmp"]),float(coord_min_max["max_msl"])]

    return (vmin), (vmax)

####################################################################################################################################

####################################################################################################################################


def plot_ar6_SLR_global_projection(ds, target_time, sample_index, bbox, lon, lat, vmin, vmax,  glob_rep, variable, cmap, target_lat, target_lon, verb):
    if glob_rep=="gridded":

        # Select the specific time and sample from the dataset
        waterlevel_at_time = ds[variable].sel(years=target_time, samples=sample_index)

        # Get latitudes and longitudes
        station_x = ds['lon'].values
        station_y = ds['lat'].values

        # Create a meshgrid for the latitudes and longitudes
        lon_grid, lat_grid = np.meshgrid(station_x, station_y)

        # Transform from EPSG:4326 to EPSG:3857
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        station_x_3857, station_y_3857 = transformer.transform(lon_grid, lat_grid)
        lon_3857, lat_3857 = transformer.transform(lon, lat)

        # Define bounding box (for Mercator projection from WGS84, global bbox = [-20000000, 20000000, -20000000, 20000000])
        # bbox = [station_x_3857.min(), station_x_3857.max(), station_y_3857.min(), station_y_3857.max()]

        # Select cartopy projection
        projection = ccrs.epsg(3857)

        # Plot setup
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent(bbox, crs=projection)

        # Add layers
        land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
        ax.add_feature(land)
        ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='dimgray', zorder=3)

        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
        title = variable + f' at {target_time}'
        ax.set_title(title, loc='left', fontsize=15)

        # Plot the data
        # scatter = ax.scatter(station_x_3857, station_y_3857, c=waterlevel_at_time.values.flatten(), cmap=cmap, marker='o', s=5, zorder=1)
        # plt.colorbar(scatter, ax=ax)

        scatter = ax.scatter(station_x_3857, station_y_3857,c=waterlevel_at_time.values.flatten(), label='Water Level', cmap=cmap,vmin=vmin, vmax=vmax, marker='o', s=5,zorder=1)
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        ax.scatter(lon_3857, lat_3857, facecolor='none', edgecolor='#990000', linewidths=2, s=100, marker='s', label=f'Beira', zorder=4)

        plt.xlabel('Longitude (EPSG:3857)')
        plt.ylabel('Latitude (EPSG:3857)')
        # Move legend outside the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05), fontsize=10)  
        plt.grid(True)

        if verb==1:
            # Add an inset for zoomed area
            zoom_factor = 0.05 # Adjust zoom factor as needed
            zoom_factor2 = 0.02 # Adjust zoom factor as needed
            left, bottom, width, height = [0.5, 0., 0.3, 0.3]
            inset_ax = fig.add_axes([left, bottom, width, height], projection=projection)

            # Box extend
            inset_bbox = [
                lon_3857 - zoom_factor2 * (bbox[1] - bbox[0]),
                lon_3857 + zoom_factor * (bbox[1] - bbox[0]),
                lat_3857 - zoom_factor * (bbox[3] - bbox[2]),
                lat_3857 + zoom_factor2 * (bbox[3] - bbox[2])
            ]
            inset_ax.set_extent(inset_bbox, crs=projection)

            # Add layers
            land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
            inset_ax.add_feature(land)
            # inset_ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='black', zorder=3)
            inset_ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='dimgray', zorder=3)
            inset_ax.add_feature(cfea.RIVERS,lw=1.5, zorder=3)
            inset_scatter = inset_ax.scatter(station_x_3857, station_y_3857, c=waterlevel_at_time.values.flatten(), cmap=cmap, vmin=vmin, vmax=vmax, marker='s', s=70, zorder=1)
            inset_ax.scatter(lon_3857, lat_3857, facecolor='none', edgecolor='#990000', linewidths=2, s=10, marker='s', zorder=4)

            # Add a red border around the inset
            rect = Rectangle((left+0.01, bottom), width-0.02, height, transform=fig.transFigure,
                            edgecolor='#990000', facecolor='none', linewidth=2)
            fig.patches.append(rect)


        plt.show()

    if glob_rep=="tide-gauges":
        # Select layer and coordinates (EPSG:4326)
        waterlevel_at_time = ds[variable].sel(years=target_time, samples=sample_index).values
        station_x = ds['lon'].values
        station_y = ds['lat'].values

        # Find nearest (euclidean) coordinates index
        distances = np.sqrt((station_x - target_lon)**2 + (station_y - target_lat)**2)
        nearest_station_index = np.argmin(distances)
        station_index = nearest_station_index  

        # Transform from EPSG:4326 to EPSG:3857
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        station_x_3857, station_y_3857 = transformer.transform(station_x, station_y)
        lon_3857, lat_3857 = transformer.transform(lon, lat)

        # Select local station coordinates
        highlight_x = station_x_3857[station_index]
        highlight_y = station_y_3857[station_index]
        highlight_value = waterlevel_at_time[station_index]

        # Define bounding box (for Mercatore projection from WGS84, global bbox = [-20000000, 20000000, -20000000, 20000000])
        bbox = [station_x_3857.min(), station_x_3857.max(), station_y_3857.min(), station_y_3857.max()]

        # Select cartopy projection
        projection = ccrs.epsg(3857)

        # Plot setup
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent(bbox, crs=projection)

        # Add layers
        land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
        ax.add_feature(land)
        ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='dimgray', zorder=3)

        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
        title = variable + f' at {target_time}'
        ax.set_title(title, loc='left', fontsize=15)

        scatter = ax.scatter(station_x_3857, station_y_3857, c=waterlevel_at_time, label='Water Level', cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=5, zorder=5)
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        ax.scatter(lon_3857, lat_3857, facecolor='none', edgecolor='#990000', linewidths=2, s=100, marker='s', label=f'Beira (nearest Station {station_index})', zorder=4)

        plt.xlabel('Longitude (degrees_east)')
        plt.ylabel('Latitude (degrees_north)')
        # Move legend outside the plot
        ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05), fontsize=10)  
        plt.grid(True)

        if verb==1:
            # Add an inset for zoomed area
            zoom_factor = 0.05 # Adjust zoom factor as needed
            zoom_factor2 = 0.02 # Adjust zoom factor as needed
            left, bottom, width, height = [0.5, 0., 0.3, 0.3]
            inset_ax = fig.add_axes([left, bottom, width, height], projection=projection)

            # Box extend
            inset_bbox = [
                lon_3857 - zoom_factor2 * (bbox[1] - bbox[0]),
                lon_3857 + zoom_factor * (bbox[1] - bbox[0]),
                lat_3857 - zoom_factor * (bbox[3] - bbox[2]),
                lat_3857 + zoom_factor2 * (bbox[3] - bbox[2])
            ]
            inset_ax.set_extent(inset_bbox, crs=projection)

            # Add layers
            land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
            inset_ax.add_feature(land)
            # inset_ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='black', zorder=3)
            inset_ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='dimgray', zorder=3)
            inset_ax.add_feature(cfea.RIVERS,lw=1.5, zorder=3)
            inset_scatter = inset_ax.scatter(station_x_3857, station_y_3857, c=waterlevel_at_time, cmap=cmap, vmin=vmin, vmax=vmax, marker='o', s=30, zorder=5)
            inset_ax.scatter(lon_3857, lat_3857, facecolor='none', edgecolor='#990000', linewidths=2, s=10, marker='s', zorder=4)

            # Add a red border around the inset
            rect = Rectangle((left+0.01, bottom), width-0.02, height, transform=fig.transFigure,
                            edgecolor='#990000', facecolor='none', linewidth=2)
            fig.patches.append(rect)

        plt.show()

####################################################################################################################################

####################################################################################################################################


def compute_other_quantiles(quantiles,lat,lon,wf,ssp,data_path):
        if ssp=="ssp126":
                
                # Open NetCDF
                ssp_tmp1="ssp245"
                filename = 'ar6-lsl-gridded-Lat'+str(lat)+'-Lon'+str(lon)+'-'+wf+'-'+ssp_tmp1+'.nc'
                file_path = data_path+'/FACTS_RSL_projections/'+filename
                ds_sel_beira_tmp = xr.open_dataset(file_path)

                # Create numpy array
                np_sel_beira_tmp = (ds_sel_beira_tmp.to_array()).as_numpy()

                calculated_quantiles_2=np.ones([5,9])
                for i in range(0,9):
                    calculated_quantiles_2[:,i] = np.quantile(np_sel_beira_tmp[0,:,i], quantiles[:])

                # Open NetCDF
                ssp_tmp2="ssp585"
                filename = 'ar6-lsl-gridded-Lat'+str(lat)+'-Lon'+str(lon)+'-'+wf+'-'+ssp_tmp2+'.nc'
                file_path = data_path+'/FACTS_RSL_projections/'+filename
                ds_sel_beira_tmp = xr.open_dataset(file_path)

                # Create numpy array
                np_sel_beira_tmp = (ds_sel_beira_tmp.to_array()).as_numpy()

                calculated_quantiles_3=np.ones([5,9])
                for i in range(0,9):
                    calculated_quantiles_3[:,i] = np.quantile(np_sel_beira_tmp[0,:,i], quantiles[:])


                del ds_sel_beira_tmp, np_sel_beira_tmp


        if ssp=="ssp245":
                
                # Open NetCDF
                ssp_tmp1="ssp126"
                filename = 'ar6-lsl-gridded-Lat'+str(lat)+'-Lon'+str(lon)+'-'+wf+'-'+ssp_tmp1+'.nc'
                file_path = data_path+'/FACTS_RSL_projections/'+filename
                ds_sel_beira_tmp = xr.open_dataset(file_path)

                # Create numpy array
                np_sel_beira_tmp = (ds_sel_beira_tmp.to_array()).as_numpy()

                calculated_quantiles_2=np.ones([5,9])
                for i in range(0,9):
                    calculated_quantiles_2[:,i] = np.quantile(np_sel_beira_tmp[0,:,i], quantiles[:])

                # Open NetCDF
                ssp_tmp2="ssp585"
                filename = 'ar6-lsl-gridded-Lat'+str(lat)+'-Lon'+str(lon)+'-'+wf+'-'+ssp_tmp2+'.nc'
                file_path = data_path+'/FACTS_RSL_projections/'+filename
                ds_sel_beira_tmp = xr.open_dataset(file_path)

                # Create numpy array
                np_sel_beira_tmp = (ds_sel_beira_tmp.to_array()).as_numpy()

                calculated_quantiles_3=np.ones([5,9])
                for i in range(0,9):
                    calculated_quantiles_3[:,i] = np.quantile(np_sel_beira_tmp[0,:,i], quantiles[:])


                del ds_sel_beira_tmp, np_sel_beira_tmp


        if ssp=="ssp585":
                
                # Open NetCDF
                ssp_tmp1="ssp245"
                filename = 'ar6-lsl-gridded-Lat'+str(lat)+'-Lon'+str(lon)+'-'+wf+'-'+ssp_tmp1+'.nc'
                file_path = data_path+'/FACTS_RSL_projections/'+filename
                ds_sel_beira_tmp = xr.open_dataset(file_path)

                # Create numpy array
                np_sel_beira_tmp = (ds_sel_beira_tmp.to_array()).as_numpy()

                calculated_quantiles_2=np.ones([5,9])
                for i in range(0,9):
                    calculated_quantiles_2[:,i] = np.quantile(np_sel_beira_tmp[0,:,i], quantiles[:])

                # Open NetCDF
                ssp_tmp2="ssp126"
                filename = 'ar6-lsl-gridded-Lat'+str(lat)+'-Lon'+str(lon)+'-'+wf+'-'+ssp_tmp2+'.nc'
                file_path = data_path+'/FACTS_RSL_projections/'+filename
                ds_sel_beira_tmp = xr.open_dataset(file_path)

                # Create numpy array
                np_sel_beira_tmp = (ds_sel_beira_tmp.to_array()).as_numpy()

                calculated_quantiles_3=np.ones([5,9])
                for i in range(0,9):
                    calculated_quantiles_3[:,i] = np.quantile(np_sel_beira_tmp[0,:,i], quantiles[:])


                del ds_sel_beira_tmp, np_sel_beira_tmp

        return calculated_quantiles_2, calculated_quantiles_3, ssp_tmp1, ssp_tmp2

####################################################################################################################################

####################################################################################################################################

def plot_surrogate_in_out(ds,waterlevel,surge,dt,variable,station_x,station_y,target_time,lon,lat,moon_az_alt_79_18,sun_az_alt_79_18, vmin, vmax,palette_hex_list, cmap_wind,cmap_tmp,cmap_msl):
    # VARIABLES
    u = ds[variable[0]].sel(time=target_time).values
    v = ds[variable[1]].sel(time=target_time).values
    temp = ds[variable[2]].sel(time=target_time).values
    msl = ds[variable[3]].sel(time=target_time).values

    magnitude = np.sqrt(u**2 + v**2)

    # COORDINATES and PROJECTION
    # Create a meshgrid for the latitudes and longitudes
    lon_grid,lat_grid = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    # Transform from EPSG:4326 to EPSG:3857
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon_x_3857,lat_y_3857 = transformer.transform(lon_grid, lat_grid)
    # Define bounding box (for Mercatore projection from WGS84, global bbox = [-20000000, 20000000, -20000000, 20000000])
    bbox = [lon_x_3857.min(), lon_x_3857.max(), lat_y_3857.min(), lat_y_3857.max()]

    # Use the sel method with method='nearest' from ERA5 Dataset
    nearest_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
    # Select local station coordinates
    highlight_x = nearest_point['longitude'].values
    highlight_y = nearest_point['latitude'].values
    highlight_x,highlight_y = transformer.transform(highlight_x, highlight_y)

    # Use the nearest euclidean distance for GTSM dataset
    distances = np.sqrt((station_x - lon)**2 + (station_y - lat)**2)
    nearest_station_index = np.argmin(distances)
    station_x, station_y = transformer.transform(station_x, station_y)
    # PLOT
    ########################################################################################################################################
    #_______________________________________________________________________________________________________________________________________
    # Plot setup
    fig = plt.figure(figsize=(16, 13))
    # Select cartopy prohection
    projection = ccrs.epsg(3857)

    #_______________________________________________________________________________________________________________________________________
    ax = fig.add_subplot(2, 2, 1, projection=projection)
    ax.set_extent(bbox, projection)
    # Add layers
    # land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
    # ax.add_feature(land)
    ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='black')
    ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='k', zorder=3)
    # ax.add_feature(cfea.RIVERS,lw=.5)
    title = r'10$\mathrm{m}$'+" wind field: "+variable[0]+  r" $\times$ " +variable[1]
    ax.set_title(title, loc='left', fontsize=15)
    # scatter = ax.scatter(lon_x_3857, lat_y_3857, c=waterlevel_at_time, label='t2m', cmap=cmap_tmp, marker='s', s=17,zorder=1)
    norm = plt.Normalize(vmin=vmin[0], vmax=vmax[0])
    Q=plt.quiver(lon_x_3857, lat_y_3857, u, v, magnitude, angles='xy', scale_units='xy', scale=0.0001, cmap=cmap_wind, norm=norm, zorder=4)
    # Add colorbar
    cbar = plt.colorbar(Q, ax=ax, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r'$\|wind\|   [\frac{\mathrm{m}}{\mathrm{s}}]$', fontsize=15)  

    ax.scatter(highlight_x, highlight_y, facecolor='none', edgecolor='red',linewidths=2, s=20, marker='s', label=f'NY Station', zorder=5)
    plt.xlabel('Longitude ')
    plt.ylabel('Latitude ')
    plt.legend()
    lon_ticks = np.linspace(bbox[0], bbox[1], 5)
    lat_ticks = np.linspace(bbox[2], bbox[3], 5)
    plt.xticks(lon_ticks)
    plt.yticks(lat_ticks)


    #_______________________________________________________________________________________________________________________________________
    ax = fig.add_subplot(2, 2, 2, projection=projection)
    ax.set_extent(bbox, projection)
    # Add layers
    # land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
    # ax.add_feature(land)
    ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='black')
    ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='k', zorder=3)
    # ax.add_feature(cfea.RIVERS,lw=.5)
    title = "2m air temperature: "+ variable[2]
    ax.set_title(title, loc='left', fontsize=15)
    scatter = ax.scatter(lon_x_3857, lat_y_3857, c=temp, cmap=cmap_tmp,vmin=vmin[1],vmax=vmax[1], marker='s', s=150,zorder=1)
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("temperature  "+r'[$\mathrm{K}$]', fontsize=12)  

    ax.scatter(highlight_x, highlight_y, facecolor='none', edgecolor='red',linewidths=2, s=20, marker='s', label=f'NY Station', zorder=4)
    plt.xlabel('Longitude ')
    plt.ylabel('Latitude ')
    plt.legend()
    lon_ticks = np.linspace(bbox[0], bbox[1], 5)
    lat_ticks = np.linspace(bbox[2], bbox[3], 5)
    # plt.xticks(lon_ticks)
    # plt.yticks(lat_ticks)
    plt.xticks(lon_ticks, [f'{tick/1e6:.0f}' for tick in lon_ticks])
    plt.yticks(lat_ticks, [f'{tick/1e6:.0f}' for tick in lat_ticks])

    #_______________________________________________________________________________________________________________________________________
    ax = fig.add_subplot(2, 2, 3, projection=projection)
    ax.set_extent(bbox, projection)
    # Add layers
    # land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
    # ax.add_feature(land)
    ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='black')
    ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='k', zorder=3)
    # ax.add_feature(cfea.RIVERS,lw=.5)
    title = "surface pressure: "+variable[3]
    ax.set_title(title, loc='left', fontsize=15)
    scatter = ax.scatter(lon_x_3857, lat_y_3857, c=msl, cmap=cmap_msl,vmin=vmin[2],vmax=vmax[2], marker='s', s=150,zorder=1)
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("pressure "+r'[$\mathrm{Pa}$]', fontsize=12)  

    ax.scatter(highlight_x, highlight_y, facecolor='none', edgecolor='red',linewidths=2, s=20, marker='s', label=f'NY Station', zorder=4)
    plt.xlabel('Longitude ')
    plt.ylabel('Latitude ')
    plt.legend()
    lon_ticks = np.linspace(bbox[0], bbox[1], 5)
    lat_ticks = np.linspace(bbox[2], bbox[3], 5)
    plt.xticks(lon_ticks)
    plt.yticks(lat_ticks)
    #_______________________________________________________________________________________________________________________________________

    # Subplot 4: 
    ax4 = fig.add_subplot(2, 2, 4)
    # remove labels and box
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['top'].set_color('none')
    ax4.spines['bottom'].set_color('none')
    ax4.spines['left'].set_color('none')
    ax4.spines['right'].set_color('none')
    # time range
    start_time = str(pd.Timestamp(target_time) - pd.Timedelta(hours=24))
    end_time = str(pd.Timestamp(target_time) + pd.Timedelta(hours=24))
    # shorter time range
    start_time_sh = str(pd.Timestamp(target_time) - pd.Timedelta(hours=6))
    end_time_sh = str(pd.Timestamp(target_time) + pd.Timedelta(hours=6))

    # nset_axes for multiple subplots 

    ################# moon and sun alt and az #################

    # convert time 
    moon_az_alt_79_18['date'] = pd.to_datetime(moon_az_alt_79_18['date'])
    sun_az_alt_79_18['date'] = pd.to_datetime(sun_az_alt_79_18['date'])
    # select values within the period
    moon_filtered = moon_az_alt_79_18[(moon_az_alt_79_18['date'] >= start_time_sh) & (moon_az_alt_79_18['date'] <= end_time_sh)]
    sun_filtered = sun_az_alt_79_18[(sun_az_alt_79_18['date'] >= start_time_sh) & (sun_az_alt_79_18['date'] <= end_time_sh)]
    # extract azimut and altitude
    moon_az_alt_79_18_np = moon_filtered[['azimut', 'altitude']].to_numpy()
    sun_az_alt_79_18_np = sun_filtered[['azimut', 'altitude']].to_numpy()

    ax4_1 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.2, 0.73, 0.1, 0.3), bbox_transform=ax4.transAxes)
    ax4_1.plot(moon_filtered['date'], moon_az_alt_79_18_np[:,0], label=f'azimut',color=palette_hex_list[0])
    ax4_1.scatter(moon_az_alt_79_18[moon_az_alt_79_18['date'] == target_time]["date"].to_numpy(),moon_az_alt_79_18[moon_az_alt_79_18['date'] == target_time]["azimut"].to_numpy(),color=palette_hex_list[0])
    ax4_1.set_ylim([0,math.pi*2])
    ax4_1.set_ylabel('azimut'+" "+r'$[rad]$',fontsize=8)
    ax4_1.set_title('Moon',fontsize=8)
    ax4_1.grid(True)
    ax4_1.xaxis.set_major_locator(plt.MaxNLocator(nbins=1))
    ax4_1.xaxis.set_tick_params(labelbottom=False)

    ax4_2 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.40, 0.73, 0.1, 0.3), bbox_transform=ax4.transAxes)
    ax4_2.plot(moon_filtered['date'], moon_az_alt_79_18_np[:,1], label=f'altitude',color=palette_hex_list[0])
    ax4_2.scatter(moon_az_alt_79_18[moon_az_alt_79_18['date'] == target_time]["date"].to_numpy(),moon_az_alt_79_18[moon_az_alt_79_18['date'] == target_time]["altitude"].to_numpy(),color=palette_hex_list[0])
    ax4_2.set_ylim([-math.pi/2,math.pi/2])
    ax4_2.set_ylabel('altitude'+" "+r'$[rad]$',fontsize=8)
    ax4_2.set_title('Moon',fontsize=8)
    ax4_2.grid(True)
    ax4_2.xaxis.set_major_locator(plt.MaxNLocator(nbins=1))
    ax4_2.xaxis.set_tick_params(labelbottom=False)

    ax4_3 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.6, 0.73, 0.1, 0.3), bbox_transform=ax4.transAxes)
    ax4_3.plot(sun_filtered['date'], sun_az_alt_79_18_np[:,0], label=f'azimut',color=palette_hex_list[6])
    ax4_3.scatter(sun_az_alt_79_18[sun_az_alt_79_18['date'] == target_time]["date"].to_numpy(),sun_az_alt_79_18[sun_az_alt_79_18['date'] == target_time]["azimut"].to_numpy(),color=palette_hex_list[6])
    ax4_3.set_ylim([0,math.pi*2])
    ax4_3.set_ylabel('azimut'+" "+r'$[rad]$',fontsize=8)
    ax4_3.set_title('Sun',fontsize=8)
    ax4_3.grid(True)
    ax4_3.xaxis.set_major_locator(plt.MaxNLocator(nbins=1))
    ax4_3.xaxis.set_tick_params(labelbottom=False)

    ax4_4 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.8, 0.73, 0.1, 0.3), bbox_transform=ax4.transAxes)
    ax4_4.plot(sun_filtered['date'], sun_az_alt_79_18_np[:,1], label=f'altitude',color=palette_hex_list[6])
    ax4_4.scatter(sun_az_alt_79_18[sun_az_alt_79_18['date'] == target_time]["date"].to_numpy(),sun_az_alt_79_18[sun_az_alt_79_18['date'] == target_time]["altitude"].to_numpy(),color=palette_hex_list[6])
    ax4_4.set_ylim([-math.pi/2,math.pi/2])
    ax4_4.set_ylabel('altitude'+" "+r'$[rad]$',fontsize=8)
    ax4_4.set_title('Sun',fontsize=8)
    ax4_4.grid(True)
    ax4_4.xaxis.set_major_locator(plt.MaxNLocator(nbins=1))
    ax4_4.xaxis.set_tick_params(labelbottom=False)

    ################# Tide-only #################
    # select in time interval
    subset_tide = waterlevel.sel(time=slice(start_time, end_time))

    ax4_5 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.1, 0.0, 0.85, 0.8), bbox_transform=ax4.transAxes)
    ax4_5.plot(subset_tide['time'].values, subset_tide["waterlevel"].values, label=f'tide-only historical 10m',color=palette_hex_list[1])
    ax4_5.scatter(subset_tide.sel(time=target_time)["time"].values,subset_tide.sel(time=target_time)["waterlevel"].values,color=palette_hex_list[1])
    ax4_5.annotate(f"{dt.year}/{dt.month}/{dt.day} {dt.hour} : {dt.minute}",
                    (subset_tide.sel(time=target_time)["time"].values,subset_tide.sel(time=target_time)["waterlevel"].values),
                    textcoords="offset points",
                    xytext=(0,5),
                    fontsize=8, 
                    color=palette_hex_list[1],  
                    rotation=0,      
                    ha='right')
    ax4_5.set_ylim([-2,3])
    ax4_5.set_ylabel('waterlevel'+" "+r'$[m]$')
    ax4_5.set_title('NY local levels')
    # ax4_5.legend(loc='upper right')
    ax4_5.grid(True)
    ax4_5.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax4_5.xaxis.set_tick_params(labelbottom=False)


    ################# Surge select #################
    # select in time interval
    subset_surge = surge.sel(time=slice(start_time, end_time))

    ax4_6 = inset_axes(ax4, width="100%", height="45%", loc='lower center', bbox_to_anchor=(0.1, 0.0, 0.85, 0.8), bbox_transform=ax4.transAxes)
    ax4_6.plot(subset_surge['time'].values, subset_surge["surge"].values, label=f'tide-only historical 10m',color=palette_hex_list[2])
    ax4_6.scatter(subset_surge.sel(time=target_time)["time"].values,subset_surge.sel(time=target_time)["surge"].values,color=palette_hex_list[2])
    ax4_6.annotate(f"{dt.year}/{dt.month}/{dt.day} {dt.hour} : {dt.minute}",
                    (subset_surge.sel(time=target_time)["time"].values,subset_surge.sel(time=target_time)["surge"].values),
                    textcoords="offset points",
                    xytext=(0,5),
                    fontsize=8, 
                    color=palette_hex_list[2],  
                    rotation=0,      
                    ha='right')
    ax4_6.set_ylim([-0.0,2.5])

    ax4_6.set_ylabel('surge'+" "+r'$[m]$')
    # ax4_6.set_xlabel('date')
    # ax4_6.legend(loc='upper right')
    ax4_6.grid(True)
    ax4_6.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    #_______________________________________________________________________________________________________________________________________

    plt.suptitle(' ERA5 reanalysis forcing and storm tide '+"    "+f"{dt.year}/{dt.month}/{dt.day}   {dt.hour} : {dt.minute}", fontsize=16)
    fig.savefig('Img/ERA5_forcing.png', dpi=500)
    plt.show()


####################################################################################################################################

####################################################################################################################################

def gif_surrogate_in_out(dur,ds,waterlevel,surge,dt,variable,target_time,lon,lat,moon_az_alt_79_18,sun_az_alt_79_18, vmin, vmax,palette_hex_list, cmap_wind,cmap_tmp,cmap_msl,times,gif_title,frames):

    file_exists = os.path.isfile("Img/"+gif_title)
    if file_exists==False:

        for target_times in times:


            target_time = str(target_times["time"].values)
            # datetime
            dt64 = np.datetime64(target_time)
            dt = pd.to_datetime(dt64)
            print()

            # VARIABLES
            u = ds[variable[0]].sel(time=target_time).values
            v = ds[variable[1]].sel(time=target_time).values
            temp = ds[variable[2]].sel(time=target_time).values
            msl = ds[variable[3]].sel(time=target_time).values

            magnitude = np.sqrt(u**2 + v**2)

            # COORDINATES and PROJECTION
            # Create a meshgrid for the latitudes and longitudes
            lon_grid,lat_grid = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
            # Transform from EPSG:4326 to EPSG:3857
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            lon_x_3857,lat_y_3857 = transformer.transform(lon_grid, lat_grid)
            # Define bounding box (for Mercatore projection from WGS84, global bbox = [-20000000, 20000000, -20000000, 20000000])
            bbox = [lon_x_3857.min(), lon_x_3857.max(), lat_y_3857.min(), lat_y_3857.max()]

            # Use the sel method with method='nearest'
            nearest_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
            # Select local station coordinates
            highlight_x = nearest_point['longitude'].values
            highlight_y = nearest_point['latitude'].values
            highlight_x,highlight_y = transformer.transform(highlight_x, highlight_y)

            # PLOT
            ########################################################################################################################################
            #_______________________________________________________________________________________________________________________________________
            # Plot setup
            fig = plt.figure(figsize=(16, 13))
            # Select cartopy prohection
            projection = ccrs.epsg(3857)

            #_______________________________________________________________________________________________________________________________________
            ax = fig.add_subplot(2, 2, 1, projection=projection)
            ax.set_extent(bbox, projection)
            # Add layers
            # land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
            # ax.add_feature(land)
            ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='black')
            ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='k', zorder=3)
            # ax.add_feature(cfea.RIVERS,lw=.5)
            title = r'10$\mathrm{m}$'+" wind field: "+variable[0]+  r" $\times$ " +variable[1]
            ax.set_title(title, loc='left', fontsize=15)
            # scatter = ax.scatter(lon_x_3857, lat_y_3857, c=waterlevel_at_time, label='t2m', cmap=cmap_tmp, marker='s', s=17,zorder=1)
            norm = plt.Normalize(vmin=vmin[0], vmax=vmax[0])
            Q=plt.quiver(lon_x_3857, lat_y_3857, u, v, magnitude, angles='xy', scale_units='xy', scale=0.0001, cmap=cmap_wind, norm=norm, zorder=4)
            # Add colorbar
            cbar = plt.colorbar(Q, ax=ax, fraction=0.02, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(r'$\|wind\|   [\frac{\mathrm{m}}{\mathrm{s}}]$', fontsize=15)  

            ax.scatter(highlight_x, highlight_y, facecolor='none', edgecolor='red',linewidths=2, s=20, marker='s', label=f'NY Station', zorder=5)
            plt.xlabel('Longitude ')
            plt.ylabel('Latitude ')
            plt.legend()
            lon_ticks = np.linspace(bbox[0], bbox[1], 5)
            lat_ticks = np.linspace(bbox[2], bbox[3], 5)
            plt.xticks(lon_ticks)
            plt.yticks(lat_ticks)


            #_______________________________________________________________________________________________________________________________________
            ax = fig.add_subplot(2, 2, 2, projection=projection)
            ax.set_extent(bbox, projection)
            # Add layers
            # land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
            # ax.add_feature(land)
            ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='black')
            ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='k', zorder=3)
            # ax.add_feature(cfea.RIVERS,lw=.5)
            title = "2m air temperature: "+ variable[2]
            ax.set_title(title, loc='left', fontsize=15)
            scatter = ax.scatter(lon_x_3857, lat_y_3857, c=temp, cmap=cmap_tmp,vmin=vmin[1],vmax=vmax[1], marker='s', s=150,zorder=1)
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("temperature  "+r'[$\mathrm{K}$]', fontsize=12)  

            ax.scatter(highlight_x, highlight_y, facecolor='none', edgecolor='red',linewidths=2, s=20, marker='s', label=f'NY Station', zorder=4)
            plt.xlabel('Longitude ')
            plt.ylabel('Latitude ')
            plt.legend()
            lon_ticks = np.linspace(bbox[0], bbox[1], 5)
            lat_ticks = np.linspace(bbox[2], bbox[3], 5)
            plt.xticks(lon_ticks, [f'{tick/1e6:.0f}' for tick in lon_ticks])
            plt.yticks(lat_ticks, [f'{tick/1e6:.0f}' for tick in lat_ticks])

            #_______________________________________________________________________________________________________________________________________
            ax = fig.add_subplot(2, 2, 3, projection=projection)
            ax.set_extent(bbox, projection)
            # Add layers
            # land = cfea.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='white', zorder=2)
            # ax.add_feature(land)
            ax.add_feature(cfea.COASTLINE, lw=1, edgecolor='black')
            ax.add_feature(cfea.BORDERS, linewidth=0.6, edgecolor='k', zorder=3)
            # ax.add_feature(cfea.RIVERS,lw=.5)
            title = "surface pressure: "+variable[3]
            ax.set_title(title, loc='left', fontsize=15)
            scatter = ax.scatter(lon_x_3857, lat_y_3857, c=msl, cmap=cmap_msl,vmin=vmin[2],vmax=vmax[2], marker='s', s=150,zorder=1)
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("pressure "+r'[$\mathrm{Pa}$]', fontsize=12)  

            ax.scatter(highlight_x, highlight_y, facecolor='none', edgecolor='red',linewidths=2, s=20, marker='s', label=f'NY Station', zorder=4)
            plt.xlabel('Longitude ')
            plt.ylabel('Latitude ')
            plt.legend()
            lon_ticks = np.linspace(bbox[0], bbox[1], 5)
            lat_ticks = np.linspace(bbox[2], bbox[3], 5)
            plt.xticks(lon_ticks)
            plt.yticks(lat_ticks)
            #_______________________________________________________________________________________________________________________________________

            # Subplot 4: 
            ax4 = fig.add_subplot(2, 2, 4)
            # remove labels and box
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.spines['top'].set_color('none')
            ax4.spines['bottom'].set_color('none')
            ax4.spines['left'].set_color('none')
            ax4.spines['right'].set_color('none')
            # time range
            start_time = str(pd.Timestamp(target_time) - pd.Timedelta(hours=24))
            end_time = str(pd.Timestamp(target_time) + pd.Timedelta(hours=24))
            # shorter time range
            start_time_sh = str(pd.Timestamp(target_time) - pd.Timedelta(hours=6))
            end_time_sh = str(pd.Timestamp(target_time) + pd.Timedelta(hours=6))

            # nset_axes for multiple subplots 

            ################# moon and sun alt and az #################

            # convert time 
            moon_az_alt_79_18['date'] = pd.to_datetime(moon_az_alt_79_18['date'])
            sun_az_alt_79_18['date'] = pd.to_datetime(sun_az_alt_79_18['date'])
            # select values within the period
            moon_filtered = moon_az_alt_79_18[(moon_az_alt_79_18['date'] >= start_time_sh) & (moon_az_alt_79_18['date'] <= end_time_sh)]
            sun_filtered = sun_az_alt_79_18[(sun_az_alt_79_18['date'] >= start_time_sh) & (sun_az_alt_79_18['date'] <= end_time_sh)]
            # extract azimut and altitude
            moon_az_alt_79_18_np = moon_filtered[['azimut', 'altitude']].to_numpy()
            sun_az_alt_79_18_np = sun_filtered[['azimut', 'altitude']].to_numpy()

            ax4_1 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.2, 0.73, 0.1, 0.3), bbox_transform=ax4.transAxes)
            ax4_1.plot(moon_filtered['date'], moon_az_alt_79_18_np[:,0], label=f'azimut',color=palette_hex_list[0])
            ax4_1.scatter(moon_az_alt_79_18[moon_az_alt_79_18['date'] == target_time]["date"].to_numpy(),moon_az_alt_79_18[moon_az_alt_79_18['date'] == target_time]["azimut"].to_numpy(),color=palette_hex_list[0])
            ax4_1.set_ylim([0,math.pi*2])
            ax4_1.set_ylabel('azimut'+" "+r'$[rad]$',fontsize=8)
            ax4_1.set_title('Moon',fontsize=8)
            ax4_1.grid(True)
            ax4_1.xaxis.set_major_locator(plt.MaxNLocator(nbins=1))
            ax4_1.xaxis.set_tick_params(labelbottom=False)

            ax4_2 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.40, 0.73, 0.1, 0.3), bbox_transform=ax4.transAxes)
            ax4_2.plot(moon_filtered['date'], moon_az_alt_79_18_np[:,1], label=f'altitude',color=palette_hex_list[0])
            ax4_2.scatter(moon_az_alt_79_18[moon_az_alt_79_18['date'] == target_time]["date"].to_numpy(),moon_az_alt_79_18[moon_az_alt_79_18['date'] == target_time]["altitude"].to_numpy(),color=palette_hex_list[0])
            ax4_2.set_ylim([-math.pi/2,math.pi/2])
            ax4_2.set_ylabel('altitude'+" "+r'$[rad]$',fontsize=8)
            ax4_2.set_title('Moon',fontsize=8)
            ax4_2.grid(True)
            ax4_2.xaxis.set_major_locator(plt.MaxNLocator(nbins=1))
            ax4_2.xaxis.set_tick_params(labelbottom=False)

            ax4_3 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.6, 0.73, 0.1, 0.3), bbox_transform=ax4.transAxes)
            ax4_3.plot(sun_filtered['date'], sun_az_alt_79_18_np[:,0], label=f'azimut',color=palette_hex_list[6])
            ax4_3.scatter(sun_az_alt_79_18[sun_az_alt_79_18['date'] == target_time]["date"].to_numpy(),sun_az_alt_79_18[sun_az_alt_79_18['date'] == target_time]["azimut"].to_numpy(),color=palette_hex_list[6])
            ax4_3.set_ylim([0,math.pi*2])
            ax4_3.set_ylabel('azimut'+" "+r'$[rad]$',fontsize=8)
            ax4_3.set_title('Sun',fontsize=8)
            ax4_3.grid(True)
            ax4_3.xaxis.set_major_locator(plt.MaxNLocator(nbins=1))
            ax4_3.xaxis.set_tick_params(labelbottom=False)

            ax4_4 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.8, 0.73, 0.1, 0.3), bbox_transform=ax4.transAxes)
            ax4_4.plot(sun_filtered['date'], sun_az_alt_79_18_np[:,1], label=f'altitude',color=palette_hex_list[6])
            ax4_4.scatter(sun_az_alt_79_18[sun_az_alt_79_18['date'] == target_time]["date"].to_numpy(),sun_az_alt_79_18[sun_az_alt_79_18['date'] == target_time]["altitude"].to_numpy(),color=palette_hex_list[6])
            ax4_4.set_ylim([-math.pi/2,math.pi/2])
            ax4_4.set_ylabel('altitude'+" "+r'$[rad]$',fontsize=8)
            ax4_4.set_title('Sun',fontsize=8)
            ax4_4.grid(True)
            ax4_4.xaxis.set_major_locator(plt.MaxNLocator(nbins=1))
            ax4_4.xaxis.set_tick_params(labelbottom=False)

            ################# Tide-only #################
            # select in time interval
            subset_tide = waterlevel.sel(time=slice(start_time, end_time))

            ax4_5 = inset_axes(ax4, width="100%", height="45%", loc='upper center', bbox_to_anchor=(0.1, 0.0, 0.85, 0.8), bbox_transform=ax4.transAxes)
            ax4_5.plot(subset_tide['time'].values, subset_tide["waterlevel"].values, label=f'tide-only historical 10m',color=palette_hex_list[1])
            ax4_5.scatter(subset_tide.sel(time=target_time)["time"].values,subset_tide.sel(time=target_time)["waterlevel"].values,color=palette_hex_list[1])
            ax4_5.annotate(f"{dt.year}/{dt.month}/{dt.day} {dt.hour} : {dt.minute}",
                            (subset_tide.sel(time=target_time)["time"].values,subset_tide.sel(time=target_time)["waterlevel"].values),
                            textcoords="offset points",
                            xytext=(0,5),
                            fontsize=8, 
                            color=palette_hex_list[1],  
                            rotation=0,      
                            ha='right')
            ax4_5.set_ylim([-2,3])
            ax4_5.set_ylabel('waterlevel'+" "+r'$[m]$')
            ax4_5.set_title('New York GTSM levels')
            # ax4_5.legend(loc='upper right')
            ax4_5.grid(True)
            ax4_5.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            ax4_5.xaxis.set_tick_params(labelbottom=False)


            ################# Surge select #################
            # select in time interval
            subset_surge = surge.sel(time=slice(start_time, end_time))

            ax4_6 = inset_axes(ax4, width="100%", height="45%", loc='lower center', bbox_to_anchor=(0.1, 0.0, 0.85, 0.8), bbox_transform=ax4.transAxes)
            ax4_6.plot(subset_surge['time'].values, subset_surge["surge"].values, label=f'tide-only historical 10m',color=palette_hex_list[2])
            ax4_6.scatter(subset_surge.sel(time=target_time)["time"].values,subset_surge.sel(time=target_time)["surge"].values,color=palette_hex_list[2])
            ax4_6.annotate(f"{dt.year}/{dt.month}/{dt.day} {dt.hour} : {dt.minute}",
                            (subset_surge.sel(time=target_time)["time"].values,subset_surge.sel(time=target_time)["surge"].values),
                            textcoords="offset points",
                            xytext=(0,5),
                            fontsize=8, 
                            color=palette_hex_list[2],  
                            rotation=0,      
                            ha='right')
            ax4_6.set_ylim([-1.5,2.5])

            ax4_6.set_ylabel('surge'+" "+r'$[m]$')
            # ax4_6.set_xlabel('date')
            # ax4_6.legend(loc='upper right')
            ax4_6.grid(True)
            ax4_6.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            #_______________________________________________________________________________________________________________________________________

            plt.suptitle(' ERA5 reanalysis forcing and storm tide '+"    "+f"{dt.year}/{dt.month}/{dt.day}   {dt.hour} : {dt.minute}", fontsize=16)

            # Salva il frame in memoria
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            plt.close(fig)

        # save GIF
        imageio.mimsave("Img/"+gif_title, frames, duration=dur)



####################################################################################################################################

####################################################################################################################################

#________________________________________________________________________________________________________________________
# select_ERA5 = which_of_ERA5_vars(0-3) * number_of_ERA5_var(4) + number_of_ERA5_var(custom)-1(for last values, the more recent. -n for others)
# reshaped_value = reshaped_array[batch_number, select_ERA5, lat, lon]

# top_of_ERA5 = which_of_ERA5_vars(3) * number_of_ERA5_var(4) + number_of_ERA5_var(custom)
# select_moonsun = top_of_ERA5 + which_of_ms_vars(0-3) * number_of_ms_var(4) + number_of_ms_var(custom)-1(for last values, the more recent. -n for others)
# reshaped_value = reshaped_array[batch_number, select_ERA5, lat, lon]
#________________________________________________________________________________________________________________________


def generator_batch_plot(items,n_of_items,LSTM_recurrent_steps,palette_hex_list,ms_recurrent_steps,cmap_gen_plot):
    fig = plt.figure(figsize=(16, 13))

    # COORDINATES and PROJECTION
    # Create a meshgrid for the latitudes and longitudes
    lat_flat = items[0,-1,:,:].ravel()
    lon_flat = items[0,-2,:,:].ravel()

    lon_grid,lat_grid = np.meshgrid(np.unique(lat_flat), np.unique(lon_flat))
    # Transform from EPSG:4326 to EPSG:3857
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon_x_3857,lat_y_3857 = transformer.transform(lon_grid, lat_grid)
    # Define bounding box (for Mercatore projection from WGS84, global bbox = [-20000000, 20000000, -20000000, 20000000])
    bbox = [(lon_x_3857.min()-10000), (lon_x_3857.max()+10000), (lat_y_3857.min()-10000), (lat_y_3857.max()+10000)]
    


    #(1)
    ax = fig.add_subplot(5, 2, 1)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
            u10 = items[0:n_of_items,0*LSTM_recurrent_steps+LSTM_recurrent_steps-1,k,l]
            # color_index = ((k+1)*l)/(items.shape[3]*items.shape[2])
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(u10,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel(r' $wind  [\frac{\mathrm{m}}{\mathrm{s}}]$', fontsize=15)  

    #(2)
    ax = fig.add_subplot(5, 2, 2)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
            v10 = items[0:n_of_items,1*LSTM_recurrent_steps+LSTM_recurrent_steps-1,k,l]
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(v10,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel(r' $wind  [\frac{\mathrm{m}}{\mathrm{s}}]$', fontsize=15)  

    #(3)
    ax = fig.add_subplot(5, 2, 3)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
            t2m = items[0:n_of_items,2*LSTM_recurrent_steps+LSTM_recurrent_steps-1,k,l]
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(t2m,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel("temperature  "+r'[$\mathrm{K}$]', fontsize=15)  

    #(4)
    ax = fig.add_subplot(5, 2, 4)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
            msl = items[0:n_of_items,3*LSTM_recurrent_steps+LSTM_recurrent_steps-1,k,l]
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(msl,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel("pressure "+r'[$\mathrm{Pa}$]', fontsize=15)  

    #(5)
    ax = fig.add_subplot(5, 2, 5)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            maz = items[0:n_of_items,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps-1),k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(maz,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel('moon azimut'+" "+r'$[rad]$',fontsize=15)

    #(6)
    ax = fig.add_subplot(5, 2, 6)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            malt = items[0:n_of_items,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps-1),k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(malt,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel('moon altitude'+" "+r'$[rad]$',fontsize=15)

    #(7)
    ax = fig.add_subplot(5, 2, 7)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            saz = items[0:n_of_items,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps-1),k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(saz,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel('sun azimut'+" "+r'$[rad]$',fontsize=15)

    #(8)
    ax = fig.add_subplot(5, 2, 8)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            salt = items[0:n_of_items,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps-1),k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(salt,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel('sun altitude'+" "+r'$[rad]$',fontsize=15)

    #(9)
    projection = ccrs.epsg(3857)
    ax = fig.add_subplot(5, 2, 9, projection=projection)
    ax.set_extent(bbox, projection)
    # Add layers
    ax.add_feature(cfea.COASTLINE, lw=3, edgecolor='black')
    lat_flat = items[0,-1,:,:].ravel()
    lon_flat = items[0,-2,:,:].ravel()
    num_points = len(lat_flat)
    indices = np.linspace(0, 1, num_points)
    colors = cmap_gen_plot(indices)
    # extended_colors = (palette_hex_list * (num_points // len(palette_hex_list) + 1))[:num_points]
    ax.scatter(lon_x_3857, lat_y_3857, c=colors, s=50, marker='s', zorder=1)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    #(10)
    ax = fig.add_subplot(5, 2, 10)
    ax.text(0.5, 0.5, 'mean sea level = '+str(items[0,-3,0,0]), fontsize=12, ha='center')
    # remove labels and box
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.suptitle('Full batch representation of item at time t', fontsize=16)
    plt.show()

####################################################################################################################################

####################################################################################################################################

#________________________________________________________________________________________________________________________
# select_ERA5 = which_of_ERA5_vars(0-3) * number_of_ERA5_var(4) + number_of_ERA5_var(custom)-1(for last values, the more recent. -n for others)
# reshaped_value = reshaped_array[batch_number, select_ERA5, lat, lon]

# top_of_ERA5 = which_of_ERA5_vars(3) * number_of_ERA5_var(4) + number_of_ERA5_var(custom)
# select_moonsun = top_of_ERA5 + which_of_ms_vars(0-3) * number_of_ms_var(4) + number_of_ms_var(custom)-1(for last values, the more recent. -n for others)
# reshaped_value = reshaped_array[batch_number, select_ERA5, lat, lon]
#________________________________________________________________________________________________________________________


def generator_oneinput_plot(items,n_input,LSTM_recurrent_steps,palette_hex_list,ms_recurrent_steps,cmap_gen_plot):
    
    fig = plt.figure(figsize=(16, 13))

    time_vector_ERA5 = ["t"]
    n = LSTM_recurrent_steps
    for i in range(1, n):
        time_vector_ERA5.append(f"t-{i}")
    time_vector_ERA5.reverse()
    time_vector_ms = ["t"]
    n = ms_recurrent_steps
    for i in range(1, n):
        time_vector_ms.append(f"t-{i}")
    time_vector_ms.reverse()



    # COORDINATES and PROJECTION
    # Create a meshgrid for the latitudes and longitudes
    lat_flat = items[0,-1,:,:].ravel()
    lon_flat = items[0,-2,:,:].ravel()

    lon_grid,lat_grid = np.meshgrid(np.unique(lat_flat), np.unique(lon_flat))
    # Transform from EPSG:4326 to EPSG:3857
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon_x_3857,lat_y_3857 = transformer.transform(lon_grid, lat_grid)
    # Define bounding box (for Mercatore projection from WGS84, global bbox = [-20000000, 20000000, -20000000, 20000000])
    bbox = [(lon_x_3857.min()-10000), (lon_x_3857.max()+10000), (lat_y_3857.min()-10000), (lat_y_3857.max()+10000)]
    


    #(1)
    ax = fig.add_subplot(5, 2, 1)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            u10 = items[n_input,0:0*LSTM_recurrent_steps+LSTM_recurrent_steps,k,l]
  
            # color_index = ((k+1)*l)/(items.shape[3]*items.shape[2])
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(u10,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel(r' $wind  [\frac{\mathrm{m}}{\mathrm{s}}]$', fontsize=15)  
    ax.set_xticks(range(LSTM_recurrent_steps))
    ax.set_xticklabels(time_vector_ERA5)

    #(2)
    ax = fig.add_subplot(5, 2, 2)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            v10 = items[n_input,0*LSTM_recurrent_steps+LSTM_recurrent_steps:1*LSTM_recurrent_steps+LSTM_recurrent_steps,k,l]   
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(v10,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel(r' $wind  [\frac{\mathrm{m}}{\mathrm{s}}]$', fontsize=15)  
    ax.set_xticks(range(LSTM_recurrent_steps))
    ax.set_xticklabels(time_vector_ERA5)

    #(3)
    ax = fig.add_subplot(5, 2, 3)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            t2m = items[n_input,1*LSTM_recurrent_steps+LSTM_recurrent_steps:2*LSTM_recurrent_steps+LSTM_recurrent_steps,k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(t2m,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel("temperature  "+r'[$\mathrm{K}$]', fontsize=15)  
    ax.set_xticks(range(LSTM_recurrent_steps))
    ax.set_xticklabels(time_vector_ERA5)

    #(4)
    ax = fig.add_subplot(5, 2, 4)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            msl = items[n_input,2*LSTM_recurrent_steps+LSTM_recurrent_steps:3*LSTM_recurrent_steps+LSTM_recurrent_steps,k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(msl,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel("pressure "+r'[$\mathrm{Pa}$]', fontsize=15)  
    ax.set_xticks(range(LSTM_recurrent_steps))
    ax.set_xticklabels(time_vector_ERA5)

    #(5)
    ax = fig.add_subplot(5, 2, 5)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            maz = items[n_input,3*LSTM_recurrent_steps+LSTM_recurrent_steps:(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps),k,l]  
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(maz,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel('moon azimut'+" "+r'$[rad]$',fontsize=15)
    ax.set_xticks(range(ms_recurrent_steps))
    ax.set_xticklabels(time_vector_ms)

    #(6)
    ax = fig.add_subplot(5, 2, 6)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            malt = items[n_input,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(0*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps),k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(malt,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel('moon altitude'+" "+r'$[rad]$',fontsize=15)
    ax.set_xticks(range(ms_recurrent_steps))
    ax.set_xticklabels(time_vector_ms)

    #(7)
    ax = fig.add_subplot(5, 2, 7)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            saz = items[n_input,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(1*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps),k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(saz,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel('sun azimut'+" "+r'$[rad]$',fontsize=15)
    ax.set_xticks(range(ms_recurrent_steps))
    ax.set_xticklabels(time_vector_ms)

    #(8)
    ax = fig.add_subplot(5, 2, 8)
    iter=0
    for k in range(items.shape[2]):
        for l in range(items.shape[3]):
  
            salt = items[n_input,(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(2*ms_recurrent_steps+ms_recurrent_steps):(3*LSTM_recurrent_steps+LSTM_recurrent_steps)+(3*ms_recurrent_steps+ms_recurrent_steps),k,l]
  
            indices = np.linspace(0, 1, items.shape[3]*items.shape[2])
            colors = cmap_gen_plot(indices)
            ax.plot(salt,color=colors[iter], label=f'Trajectory {k},{l}')
            iter=iter+1
    ax.set_ylabel('sun altitude'+" "+r'$[rad]$',fontsize=15)
    ax.set_xticks(range(ms_recurrent_steps))
    ax.set_xticklabels(time_vector_ms)

    #(9)
    projection = ccrs.epsg(3857)
    ax = fig.add_subplot(5, 2, 9, projection=projection)
    ax.set_extent(bbox, projection)
    # Add layers
    ax.add_feature(cfea.COASTLINE, lw=3, edgecolor='black')
    lat_flat = items[0,-1,:,:].ravel()
    lon_flat = items[0,-2,:,:].ravel()
    num_points = len(lat_flat)
    indices = np.linspace(0, 1, num_points)
    colors = cmap_gen_plot(indices)
    # extended_colors = (palette_hex_list * (num_points // len(palette_hex_list) + 1))[:num_points]
    ax.scatter(lon_x_3857, lat_y_3857, c=colors, s=50, marker='s', zorder=1)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    #(10)
    ax = fig.add_subplot(5, 2, 10)
    ax.text(0.5, 0.5, 'mean sea level = '+str(items[0,-3,0,0]), fontsize=12, ha='center')
    # remove labels and box
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.suptitle('Full batch representation of item at time t', fontsize=16)
    plt.show()


