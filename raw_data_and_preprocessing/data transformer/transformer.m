% image transformer

%{
based on the mhd reading function by UVa course: Machine Learning in Image Analysis

note: you need to install Mapping Toolbox to run this program.

1. place two .m files in the root dictionary of data. (together with .mhd/.raw file)
2. run transformer.m
3. a new folder will be made and hold all .png file generated.
%}

clear all

% read all .mhd file and save their name
fileList = dir('*.mhd');
nameCell = extractfield(fileList, 'name');
length = size(nameCell, 2);

% make new folder to hold generated image
mkdir('data')

% generate image
for i = 1:length
    nameCell{i} = append('.\', nameCell{i});
    img = loadMETA(nameCell{i});
    imwrite(img, append('.\data\',string(i),'.png'));
end











