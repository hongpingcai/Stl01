%% Added by Hongping Cai
% Stl01_cluster.m
% Using pretrained Caffenet features do clustering, compare the DEC
% performance in the DEC paper.
%
% STEP1: generate the pre-trained caffenet features on stl10 
% STEP2: Kmeans clustering
% STEP3: evaluation performance
%

%%

Stl01_init;
cluster_method = 'kmeans';
cluster_K = 10;
feat_type = 'caffenet';
feat_blob = 'fc7';
feat_dim  = 128;
train_dataset = 'imagenet';%'tless';%
caffe_input_w = 227; 
caffe_input_h = 227;
be_show = 0;

%% load the testing image and label
test_mat = fullfile(dir_dataset,'test.mat');
load(test_mat,'X','y','class_names');
test_patches = reshape(X,size(X,1),96,96,3);
test_labels  = y;
n_im_test = length(test_labels);

switch lower(train_dataset)
    case 'tless'
        opt_split_trainval = 3;
        dir_Tless01 = fullfile(dir_DATA,'Hongping/Tless01');
        net_prototxt = fullfile(dir_Tless01,'caffenet-prototxt','deploy.prototxt');
        net_caffemodel = fullfile(dir_Tless01,'caffenet-model',...
            ['m' int2str(opt_split_trainval) '_Tless-caffenet_iter_10000.caffemodel']);
    case 'imagenet'
        net_prototxt = fullfile(dir_DATA,'Hongping/model-caffenet/deploy.prototxt');
        net_caffemodel = fullfile(dir_DATA,'Hongping/model-caffenet/bvlc_reference_caffenet.caffemodel');
    otherwise
        error('No such train_dataset.');
end;

%% generate the testing features
mat_test_feat = fullfile(dir_Stl01, ['test_feat.mat']);
if ~exist(mat_test_feat,'file')
    disp('** Generate the test features....');
    mat_mean = fullfile(dir_DATA,'Hongping/Tless02/ilsvrc_2012_mean_227.mat');
    caffe.set_mode_gpu();
    gpu_id = 0;  % we will use the first gpu in this demo
    caffe.set_device(gpu_id);
    net = caffe.Net(net_prototxt, net_caffemodel,'test');
    test_feats = zeros(n_im_test,4096);
    for i=1:n_im_test
        if mod(i,50)==1
            fprintf(1,'%d ',i);
        end;
        im = squeeze(test_patches(i,:,:,:));
        input_data = {prepare_image(im,caffe_input_w,caffe_input_h,mat_mean)};
        scores = net.forward(input_data);
        cur_feat = net.blobs(feat_blob).get_data();
        test_feats(i,:) = cur_feat';%%%%%%%%
    end;
    fprintf(1,'\n  Save features into %s\n',mat_test_feat);
    caffe.reset_all();
    save(mat_test_feat,'test_feats');
else
    fprintf(1,'** Load the testing features frm %s....\n',mat_test_feat);
    load(mat_test_feat,'test_feats');
end;

%% STEP3: clustering
fprintf(1, 'STEP3: clustering...\n');
mat_cluster = fullfile(dir_Stl01,[cluster_method '_' int2str(cluster_K) '.mat']);
if ~exist(mat_cluster,'file')
    rng(1); % For reproducibility
    [ids_cluster,centres_cluster,sumd,D] = kmeans(test_feats,cluster_K,'MaxIter',1000,...
        'start','cluster','Display','final','Replicates',10);    
    % find the cluster centre images
    ids_centre = zeros(1, cluster_K);
    for i=1:cluster_K
        ids_cur = find(ids_cluster == i);
        dis_cur = D(ids_cur,i);
        [v,d] = min(dis_cur);
        ids_centre(i) = ids_cur(d);
    end;
    save(mat_cluster,'ids_cluster','D','centres_cluster','ids_centre');
else
    fprintf(1,'** Load clustering file: %s ....\n',mat_cluster);
    load(mat_cluster);
end;

%% STEP4: clustering performance
disp('Cluster on stl10-test set.');
fprintf(1,'STEP4: clustering performance(pre-trained caffenet,%s)\n',feat_blob);
theta_group_purity = 0.8;
[ACC] = eval_cluster1(ids_cluster, test_labels);%
[nmi_score] = nmi(ids_cluster,double(test_labels));
[rec,pre,tp,acc_fm,tp_fm] = eval_cluster2(ids_cluster, test_labels, theta_group_purity);
fprintf(1,'** ACC: %.4f\n',ACC);
fprintf(1,'** NMI: %.4f\n',nmi_score);
fprintf(1,'** Obj-wise: rec: %.4f, pre: %.4f, tp:%d\n',rec,pre,tp);
fprintf(1,'** Frm-wise: acc_fm: %.4f, tp_fm: %d\n', acc_fm,tp_fm);

