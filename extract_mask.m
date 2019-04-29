%Extract annotation Mast to target path
Json_File_Path==#MSCOCO anntation json file path,e.g.'MSCOCO/annotations/instances_train2017.json'
Image_File_Path=#MSCOCO image file path,e.g.'MSCOCO/images/train2017'
Mask_File_Path=#MSCOCO instance mask output file path, e.g.'MSCOCO/masks/'

data=gason(fileread(Json_File_Path));
annFile=sprintf(Image_File_Path);
coco=CocoApi(annFile);
dataType =  'train2017';

for i= 1:numel(data.annotations)
    if data.annotations(i).iscrowd == 1
        continue;
    end
    img_id = coco.loadImgs(data.annotations(i).image_id);
    I = imread(sprintf('%s/%s/%s',Image_File_Path,dataType,img_id.file_name));
    category_id=data.annotations(i).category_id;
    img_name=data.annotations(i).id;
    instance_num(category_id)=+1;

    number=size(data.annotations(i).segmentation{1,1},2)/2;
    polygon = reshape(data.annotations(i).segmentation{1,1},2,number);
    c=polygon(1,:);
    r=polygon(2,:);
    BW=roipoly(I,c,r);

    mask=im2uint8(ones(size(I,1),size(I,2),3,'single'));
    mask(:,:,1)= mask(:,:,1).*uint8(~BW);
    mask(:,:,2)= mask(:,:,2).*uint8(~BW);
    mask(:,:,3)= mask(:,:,3).*uint8(~BW);
    for i=1:numel(data.categories)
        if category_id == data.categories(i).id
            cls_name = data.categories(i).name;
        end
    end

    if ~ exist(sprintf('%s/%s',Mask_File_Path,cls_name))
        mkdir(sprintf('%s/%s',Mask_File,Path,cls_name))
    end
    imwrite(mask,sprintf('%s/%s/%s.pbm',Mask_File_Path,cls_name,num2str(img_name)))
end
fprintf('Extraction done\n')
