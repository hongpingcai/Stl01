%% added by Hongping Cai, 03/05/2017
% Stl01_init.m
%%
which_server = 'deepthought';%'hc16826';%
dir_caffe = ['/home/' lower(which_server) '/Libraries/caffe'];

addpath('../common_func/');
addpath(fullfile(dir_caffe, 'matlab')); %% for caffe
addpath('../../CodesDownload/export-fig/')
addpath('../../CodesDownload/hungarian');

dir_DATA    = ['/media/deepthought/DATA'];
dir_dataset = fullfile(dir_DATA, 'Datasets/stl10/stl10_matlab');

dir_Stl01 = fullfile(dir_DATA,'Hongping/Stl01');

