p = genpath('F:\experiment\processing_diatom_database\xinyu\matlab_pr\cut_number1\');
length_p = size(p,2);
path = {};
temp = [];
for i = 1:length_p 
    if p(i) ~= ';'
        temp = [temp p(i)];
    else 
        temp = [temp '\']; 
        path = [path ; temp];
        temp = [];
    end
end  
clear p length_p temp;

%file_path =  'F:\experiment\processing_diatom_database\xinyu\matlab_pr\cut_number1\';% 图像文件夹路径
out_path =  'F:\experiment\processing_diatom_database\xinyu\matlab_pr\his_cut\';
file_num = size(path,1);
for i = 1:file_num
    file_path =  path{i}; 
    img_path_list = dir(strcat(file_path,'*.png'));
    img_num = length(img_path_list);
    if img_num > 0 
         for j = 1:img_num 
                image_name = img_path_list(j).name;
                image =  imread(strcat(file_path,image_name));
                fprintf('%d %s\n',j,strcat(file_path,image_name));
            
 
                 hist_im = imhist(image); 
                 bar(hist_im);
            
                 ima=histeq(image);
                 his_i=imhist(ima);
                 bar(his_i);
            
                 imwrite(ima,strcat(out_path,image_name));
          end
    end 
end


