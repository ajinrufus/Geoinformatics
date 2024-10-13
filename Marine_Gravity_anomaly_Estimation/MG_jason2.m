%% for all the files combined

% region taken 13 to 19 and 86 to 92

Path       = 'C:\Users\AJIN RUFUS\Documents\My Files\CE678a\Project Work 2\Jason2_Data'; % wherever you want to search
searchPath = [Path ,'\**\*.nc']; % Search in folder and subfolders for  *.nc
Files      = dir(searchPath); % Find all .nc files
%%
for i = 1:size(Files,1) % write each string in a loop
  file{i} = fullfile(Files(i).folder, Files(i).name); % write the directory of each data file
end

figure;
ax=worldmap([5 25],[70 100]); % the region of concern
load coastlines.mat % to plot the coastline boundaries
hold on 
plotm(coastlat,coastlon) 
hold on

for i = 1:size(Files(:,1))

    time{i} = {ncread(file{i},'jday.00')}; % Julian date measurements

    latitude{i} = ncread(file{i},'glat.00'); % latitude of each measurement
 
    longitude{i} = ncread(file{i},'glon.00'); % longitude of each measurement

    sea_surface_heights{i} = ncread(file{i},'ssh.33'); % sea surface height of each observation point 

    SSH_corr{i} = ncread(file{i},'sflag.33'); % sea surface correction for each observation
    
    orb_flags{i} = ncread(file{i},'oflags.00'); % orbit flags for each observation 
    
    instrument{i} = ncread(file{i},'iflags.00'); % instrument flags for each observation

    geoshow(latitude{i},longitude{i}) 
    hold on

end

%% Parsing ascending and descending pass

% the plot here is running slowly, so I have commented it.

figure;
for i = 1:length(latitude) 

    for j = 1:length(latitude{1,i}) - 1

        if latitude{1,i}(j+1,1) > latitude{1,i}(j,1) % condition to split ascending and descending track separately

            ascend_lat{i} = latitude{i}; % ascending latitude
            ascend_long{i} = longitude{i}; % ascending longitude

%             plot(ascend_long{i},ascend_lat{i})
%             hold on           
        else
            descend_lat{i} = latitude{i}; % descending latitude
            descend_long{i} = longitude{i}; % descending longitude
            
%             subplot(1,2,2)
%             plot(descend_long{i},descend_lat{i})
%             hold on
%             han=axes(fig,'visible','off'); 
%             han.Title.Visible='on';
%             han.XLabel.Visible='on';
%             han.YLabel.Visible='on';
%             title('Separation of Ascending and Descending Tracks');
%             xlabel('Longitude')
%             ylabel('Latitude')
        
        end
    end
end

%% manual method

file_name = {'\*\*_0014ssh.33.nc';'\*\*_0053ssh.33.nc';'\*\*_0090ssh.33.nc';'\*\*_0129ssh.33.nc';'\*\*_0192ssh.33.nc';'\*\*_0231ssh.33.nc'}; % file name of each pass

for i = 1:length(file_name)

    fstruct{i} = dir(strcat(Path,file_name{i})); % each pass separately as a struct
end

for i = 1:length(fstruct)
    for j = 1:length(fstruct{i})
        
        file1{j,i} = fullfile(fstruct{i}(j).folder, fstruct{i}(j).name); % directory of all the files with each pass in separate column
    end
end

%% each pass separately

for i = 1:width(file1)
    for j = 1:length(file1)
        if not(isempty(file1{j,i})) % to five when there is no empty cells
            time_per_cycle{j,i} = ncread(file1{j,i},'jday.00'); % Julian date measurements
            lat_per_cycle{j,i} = ncread(file1{j,i},'glat.00'); % latitude of each pass seprately
            lon_per_cycle{j,i} = ncread(file1{j,i},'glon.00'); % longitude of each pass seprately
            SSH_per_cycle{j,i} = ncread(file1{j,i},'ssh.33'); % sea surface heights of each pass separately

            SSH_corr_per_cycle{j,i} = ncread(file1{j,i},'sflag.33'); % sea surface correction for each observation
            orb_flags_per_cycle{j,i} = ncread(file1{j,i},'oflags.00'); % orbit flags for each observation 
            instrument_per_cycle{j,i} = ncread(file1{j,i},'iflags.00'); % instrument flags for each observation
        end
    end
end

%% to find the pass positions of satellite descending

fig = figure;
for i = 1:width(lat_per_cycle)

    for j = 1:length(lat_per_cycle{1,i}) -1

        if lat_per_cycle{1,i}(j+1,1) > lat_per_cycle{1,i}(j,1) % condition for along track values

            ascend_lat_sep{1,i} = lat_per_cycle{1,i}; % a single along track latitude
            ascend_lon_sep{1,i} = lon_per_cycle{1,i}; % a single along track latitude
            
            subplot(1,2,1)
            plot(ascend_lon_sep{i},ascend_lat_sep{i})
            hold on

        else
            descend_lat_sep{1,i} = lat_per_cycle{1,i}; % a single cross track latitude
            descend_lon_sep{1,i} = lon_per_cycle{1,i}; % a single cross track longitude
            
            subplot(1,2,2)
            plot(descend_lon_sep{i},descend_lat_sep{i})
            hold on
            han=axes(fig,'visible','off'); 
            han.Title.Visible='on';
            han.XLabel.Visible='on';
            han.YLabel.Visible='on';
            title('Single Cycle per pass of Ascending and Descending Tracks');
            xlabel('Longitude')
            ylabel('Latitude')
        end
    end
end

%% Descending passes of all the cycles

pos_descend_track = cellfun('isempty',ascend_lat_sep); % the postions of descending tracks

descend_track = find(pos_descend_track == 1); % positions of descending tracks

la111 = [];
lo111 = [];
SSH111 = [];

SSH_corr3 = [];
orb_flags3 = [];
instrument3 = [];
time3 = [];

for i = 1:length(descend_track) 

    la11 = lat_per_cycle(:,descend_track(i)); 
    lo11 = lon_per_cycle(:,descend_track(i)); 
    SSH11 = SSH_per_cycle(:,descend_track(i));
    time2 = time_per_cycle(:,descend_track(i)); 

    SSH_corr2 = SSH_corr_per_cycle(:,descend_track(i)); % sea surface correction for each observation
    orb_flags2 = orb_flags_per_cycle(:,descend_track(i)); % orbit flags for each observation 
    instrument2 = instrument_per_cycle(:,descend_track(i)); % instrument flags for each observation

    la111 = [la111,la11]; % latitude of all the cycles
    lo111 = [lo111,lo11]; % longitude of all the cycles
    SSH111 = [SSH111,SSH11]; % sea surface heights of all the cycles
    time3 = [time3,time2]; % Julain Date of all the cycles
    SSH_corr3 = [SSH_corr3,SSH_corr2]; % SSH corrections for all the cycles
    orb_flags3 = [orb_flags3,orb_flags2]; % orbit flags for all the cycles
    instrument3 = [instrument3,instrument2]; % instrument flags for all the cycles
end

%%
for i = 1:width(time3)
    for j = 1:length(time3)
            dates1{j,i} = datetime(time3{j,i}(:)+ 2451545,'convertfrom','juliandate','Format','yyy-MM'); % from julian date to YYYY-MM format
    end
end

%% variation of SSH height with time

fig =figure;
for i = 1:width(la111)
    for j = 1:length(dates1)
        subplot(4,2,i)
        plot(dates1{j,i}(:),SSH111{j,i}(:))
        hold on
    end
    han=axes(fig,'visible','off'); 
    han.Title.Visible='on';
    han.XLabel.Visible='on';
    han.YLabel.Visible='on';
    title('Sea Surface Height vs Pass');
    xlabel('Time Period')
    ylabel('Sea Surface Heights')
    
end

%% mean of each descending track cycle

% This is to find the median of all the cycles of a single pass as 
% finding the DOV components for all the cycles for a particular pass will be too complex

for i = 1:width(lo111)
    for j = 1:length(lo111)
        if not(isempty(lo111{j,i}))
            R_mean{j,i} = mean(lo111{j,i}(:)); % mean value of each of the along track cycle
        end
    end
end

%% The median cycle for each along track pass

rmean = [];
% figure;
for i = 1:width(R_mean)

        r_mean1 = cell2mat(R_mean(:,i)); % converting the cell to matrix as the function median will not work in cell            
        
        r_median(1,i) = median(r_mean1); % median of each along track pass

        mean_mat = cell2mat(R_mean(:,i)); 

        [~,nearest_pos] = min( abs( r_median(1,i) - mean_mat) );
        nearestNum = mean_mat( nearest_pos ); % to find the nearest value to the median of each pass
        nearest_index = find(mean_mat == nearestNum); % find the index of the nearest value in the 
        lat_pass{1,i} = la111{nearest_index,i}; % latitude of the median of each pass
        lon_pass{1,i} = lo111{nearest_index,i}; % longitude of the median of each pass
        SSH_pass{1,i} = SSH111{nearest_index,i}; % sea surface heights of the median of each pass
        time4{1,i} = time3{nearest_index,i};

        SSH_corr4{1,i} = SSH_corr3{nearest_index,i};
        orb_flags4{1,i} = SSH_corr3{nearest_index,i};
        instrument4{1,i} = SSH_corr3{nearest_index,i};
        
        plot(lon_pass{1,i},lat_pass{1,i})
        hold on
        xlabel('longitude')
        ylabel('Latitude')
        title('Ascending Tracks','FontWeight','bold')

end

%% Topex Poseidon Ellipsoid

customPlanet = referenceEllipsoid;
customPlanet.Name = 'T/P ellispoid';
customPlanet.LengthUnit = 'meter';
customPlanet.SemimajorAxis = 6378135.59; % length of semi major axis
customPlanet.InverseFlattening = 298.257; % inverse flattening
customPlanet.SemiminorAxis = 6356750.89294343; % length of semi minor axis
customPlanet.Eccentricity = 0.081819221455524; % first eccentricity

%% slope of the along track pass

for i = 1: length(lat_pass)
    for j = 1:length(lat_pass{i})-1

        [c{j,i},azi_a{j,i}] = distance(lat_pass{1,i}(j+1,1), lon_pass{1,i}(j+1,1), lat_pass{1,i}(j,1), lon_pass{1,i}(j,1),customPlanet,"degrees");

        height_var{j,i} = SSH_pass{1,i}(j+1,1) - SSH_pass{1,i}(j,1); % along track sea surface height variation

        slope_descend{j,i} = -(height_var{j,i}./c{j,i}); % descending track slope
        C{j,i} = 1/c{j,i}; % used to find the weight matrix
    end 
    P{i} = diag(cell2mat(C(:,i))); % weight matrix
end
%%
for i = 1: length(lat_pass)
    azimuth_of_pts{i} = cell2mat(azi_a(:,i)); % azimuth of all the observations
    data{i} = cell2mat(slope_descend(1:length(P{i}),i)); % DOV of all the observations
end

%% To form a grid

M = 13:0.25:19 ; N = 86:0.25:92; % range of latitude and longiitude

[X,Y] = meshgrid(M,N);
[Y1,X1] = meshgrid(N,M);

figure;
for i = 1:length(lon_pass)
    
    plot(lon_pass{i}(:),lat_pass{i}(:),'.r') ;
    hold on
    plot(Y,X,'k') ; hold on 
    plot(Y1,X1,'k')
    hold on
    xlabel('Longitude')
    ylabel('Latitude')
    title('Gridding of the Study Area','FontWeight','bold')
end


%% empty cells 

P_grid = cell(length(lat_pass),length(M),length(N));

grid_mid = cell(length(M) - 1,length(N) - 1);

grid_mid1 = cell(length(M) - 1,length(N) - 1);

C_grid = cell(length(M) - 1,length(N) - 1);

arclen = cell(length(M) - 1,length(N) - 1);

P_weight = cell(length(M) - 1,length(N) - 1);

XZ1 = cell(length(M) - 1,length(N) - 1);

des_A = cell(length(M) - 1,length(N) - 1);

XZ1_north = cell(length(M) - 1,length(N) - 1);

XZ1_east = cell(length(M) - 1,length(N) - 1);

for ii = 1:length(lat_pass)
    for i = 1:length(N)-1
        for j = 1:length(M)-1
        
            A = [X(i,j) Y(i,j)];
            B = [X(i+1,j+1) Y(i+1,j+1)];

            % to find the intersect of observations in each grid
            idx = find(lat_pass{ii}(1:end-1) >= A(1) & lat_pass{ii}(1:end-1) <B(1));
            idy = find(lon_pass{ii}(1:end-1) >= A(2) & lon_pass{ii}(1:end-1) <B(2)) ;
            id = intersect(idx,idy);
            P_grid{ii,i,j} = [lat_pass{ii}(id),lon_pass{ii}(id),data{ii}(id),azimuth_of_pts{ii}(id),SSH_pass{ii}(id)] ;

            if not(isempty(P_grid{ii,i,j}))
                C_grid{i,j} = P_grid{ii,i,j};
                grid_mid{i,j} = (A(1) + B(1))/2;
                grid_mid1{i,j} = (A(2) + B(2))/2;
            end

            if not(isempty(C_grid{i,j}))
                arclen{i,j} = distance(grid_mid{i,j},grid_mid1{i,j},C_grid{i,j}(:,1),C_grid{i,j}(:,2), customPlanet);
            end 
            
            P_weight{i,j} = diag(1 ./ arclen{i,j});

            if not(isempty(C_grid{i,j}))
                des_A{i,j} = [cos(deg2rad(C_grid{i,j}(:,4))) , sin(deg2rad(C_grid{i,j}(:,4)))]; % design matrix
            end
             
            if not(isempty(arclen{i,j}))
                XZ1{i,j} = inv(double(des_A{i,j}' * P_weight{i,j} * des_A{i,j})) * (des_A{i,j}' * P_weight{i,j} * C_grid{i,j}(:,3)); % DOV components
            end

            if not(isempty(arclen{i,j}))
                XZ1_north{i,j} = XZ1{i,j}(1,1); % DOV north components
                XZ1_east{i,j} = XZ1{i,j}(2,1); % DOV east components

            end
            grid_mid2{i,j} = (A(1) + B(1))/2; % mid latitude of all the points
            grid_mid3{i,j} = (A(2) + B(2))/2; % mid longitude of all the points
        end
    end
end

% since the meshgrid was done along the different axis, rotations are
% applied to bring it to the actual positions

midgrid = rot90(grid_mid); % actual latitude values of available points

midgrid1 = rot90(grid_mid1); % actual longitude values of available points

XZ1_east1 = rot90(XZ1_east); % prime DOV component

XZ1_north1 = rot90(XZ1_north); % meridional DOV component

midgrid2 = rot90(grid_mid2); % latitude of all the mid points of the grid

midgrid3 = rot90(grid_mid3); % longitude of all the mid points of the grid

%% indices of available and non avaialble points

point_available = find(~cellfun(@isempty,XZ1_north1)); % indices of available points
point_notavailable = find(cellfun(@isempty,XZ1_north1)); % indices of non available points

%% latitude and longitude of non available points

grid_lat_not = cell2mat(midgrid2(point_notavailable)); % lat of non available points
grid_lon_not = cell2mat(midgrid3(point_notavailable)); % long of non available points

%% latitude, longitude, north-south DOV components, East-West DOV components 

grid_lat = cell2mat(midgrid(point_available)); % lat of avaialable points 

grid_lon = cell2mat(midgrid1(point_available)); % lon of available points

XZ1_present = double(cell2mat(XZ1_north1(point_available))); % available north-south components

XZ2_present = double(cell2mat(XZ1_east1(point_available))); % available east-west components

%% Interpolation

rng default;
samp3 = [grid_lon,grid_lat];  % longitude, latitude of available points

figure;
scatter(grid_lon,grid_lat,'filled')
xlabel('longitude')
ylabel('latitude')
title('DOV Available Points','FontWeight','bold')

F = scatteredInterpolant(samp3(:,1),samp3(:,2),XZ1_present,'linear','linear'); % interpolation of north south components 

F2 = scatteredInterpolant(samp3(:,1),samp3(:,2),XZ2_present,'linear','linear'); % interpolation of prime DOV components

Vq = F(grid_lon_not,grid_lat_not); % interpolated north south components

Vq2 = F2(grid_lon_not,grid_lat_not); % interpolated east west components

figure;
scatter(grid_lon_not,grid_lat_not,'filled')
xlabel('longitude')
ylabel('latitude')
title('DOV Non-Available Points','FontWeight','bold')

%% placing the intrpolated values in the grid

% for north south components

grid_DOV_comp = cell(length(M) - 1,length(N) - 1);

for i = 1 : length(point_notavailable)

    grid_DOV_comp{point_notavailable(i)} = Vq(i);

end

for i = 1 : length(point_available)

    grid_DOV_comp{point_available(i)} = XZ1_present(i);

end

% for east west components

grid_DOV_comp1 = cell(length(M) - 1,length(N) - 1);

for i = 1 : length(point_notavailable)

    grid_DOV_comp1{point_notavailable(i)} = Vq2(i);

end

for i = 1 : length(point_available)

    grid_DOV_comp1{point_available(i)} = XZ2_present(i);

end

%% plotting the DOV components

DOV_comp = cell2mat(grid_DOV_comp);

DOV_comp1 = cell2mat(grid_DOV_comp1);

grid_mid_lat = cell2mat(grid_mid2);

grid_mid_lon = cell2mat(grid_mid3);

dddd = cell2mat(midgrid3);
eeee = cell2mat(midgrid2);

figure;
plot3(dddd,eeee,DOV_comp,'.')
grid on
set(gca,'Xtick',86:0.15:92)
set(gca,'Ytick',13:0.15:19)
xlabel('longitude')
ylabel('latitude')
zlabel('North South DOV Components')
title('North South DOV Components','FontWeight','bold')

figure;
plot3(dddd,eeee,DOV_comp1,'.')
grid on
set(gca,'Xtick',86:0.15:92)
set(gca,'Ytick',13:0.15:19)
xlabel('longitude')
ylabel('latitude')
zlabel('East West DOV Component')
title('East West DOV Components','FontWeight','bold')

%% normal gravity

gamma_a = 9.798; % normal gravity along the equator

gamma_b = 9.863; % normal gravity along the poles 

gamma_req = (customPlanet.SemimajorAxis .* gamma_a .* cos(deg2rad(grid_mid_lat)).^2 + customPlanet.SemiminorAxis * gamma_b * sin(deg2rad(grid_mid_lat)).^2) ./ (sqrt(customPlanet.SemimajorAxis.^2 .* cos(deg2rad(grid_mid_lon)).^2 + customPlanet.SemiminorAxis.^2 * sin(deg2rad(grid_mid_lon)).^2)); % normal gravity in the required region

figure;
imagesc(gamma_req)
colorbar
xlabel('Grid points along X- direction')
ylabel('Grid points along Y- direction')
title('Normal Gravity','FontWeight','bold')

%% to compute the wave number

Nx = 92-86; % length of spatial domain x direction
Ny = 19-13; % length of spatial domain y direction

dx = 0.25;  % Distance increment (i.e., Spacing between each column)
dy = 0.25; % Time increment (i.e., Spacing between each row)

kx = 2*pi*dx/Nx; % wave number along x -direction 

ky = 2*pi*dy/Ny; % wave number along y - direction

kk = sqrt(kx^2 + ky^2);

%% 2D FFT to compute the gravity anomaly

% since the attained DOV components are in seconds, they are converted into radians

north_south = deg2rad(DOV_comp./3600);

east_west = deg2rad(DOV_comp1./3600);

% as we were not able to find the wave number values and as it there was
% only a slight variation due to vave numbers, we neglected it

four1 = fft(north_south); % fourier transform for north component

four2 = fft(east_west); % fourier transform for east component

gravity_anomaly = abs(ifft((kx .* four1 + ky .* four2).*gamma_req .* -1i ./kk))*10^5; % gravity anomaly which is the sum of cosine and sine waves 

figure;
imagesc(gravity_anomaly)
col = colorbar;
ylabel(col,'mGal')
xlabel('Grid points along X- direction')
ylabel('Grid points along Y- direction')
title('Gravity Anomaly','FontWeight','bold')