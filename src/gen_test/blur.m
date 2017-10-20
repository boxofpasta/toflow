function blur(data_path)

scale_factor    = 4;

%% read input images
filenames   = dir(fullfile(data_path, '*.png'));
num_imgs    = length(filenames);
img_list    = cell(num_imgs, 1);
for iimg = 1 : num_imgs
  img_list{iimg} = im2double(imread(fullfile(data_path, filenames(iimg).name)));
	img_list{iimg} = dsr_imresize(img_list{iimg}, size(img_list{iimg}) / scale_factor, 'bicubic');
end

for iimg = 1 : num_imgs
	img_list{iimg} = dsr_imresize(img_list{iimg}, size(img_list{iimg}) * scale_factor, 'bicubic');
end

if ~exist([data_path '/blur'], 'dir')
	mkdir([data_path '/blur']);
end
for iimg = 1 : num_imgs
	imwrite(img_list{iimg}, [data_path '/blur/im' sprintf('%04d',iimg) '.png'])
end

end