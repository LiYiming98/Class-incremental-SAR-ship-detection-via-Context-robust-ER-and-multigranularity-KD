close all;clear ;clc;
% 获取当前文件夹中所有PNG文件
training_path = "YOUR_TRAINING_IMAGES_PATH"
pictureFiles = dir('YOUR_TRAINING_IMAGES_PATH/*.png')

test_path = "YOUR_TEST_IMAGES_PATH"
test_pictureFiles = dir('YOUR_TEST_IMAGES_PATH/*.png')
%training_path = test_path
%pictureFiles = test_pictureFiles



% 遍历所有PNG文件
for k=1: length(pictureFiles)
    tic;
    fileName = pictureFiles(k).name;
    IN=im2double(imread(training_path + fileName));  %变为双精度，即0-1

    figure;
    imhist(imread(training_path + fileName));  %axis([0 1,0,600000]);      %显示灰度直方图

    IN = IN(:,:,1);
    J=wiener2(IN);  %自适应滤波
    K=medfilt2(J);  %中值滤波

    [M,N]=size(IN);                     %得到图像行列像素
    number_all=M*N;                    %总像素值
    hui_all=0;                         %预设图像总灰度值为0
    ICV_t=0;                           %预设最大方差为0
    %得到图像总灰度值
    for i=1:M
        for j=1:N
            hui_all=hui_all+K(i,j);
        end
    end
    all_ave=hui_all*255/number_all;   %图像灰度值的总平均值
     
     
    %t为某个阈值，把原图像分为A部分（每个像素值>=t）与B部分（每个像素值<t）
    scores = zeros(1,256);
    for t=0:255                       %不断试探最优t值
        hui_A=0;                      %不断重置A部分总灰度值
        hui_B=0;                      %不断重置B部分总灰度值
        number_A=0;                   %不断重置A部分总像素
        number_B=0;                   %不断重置B部分总像素
        for i=1:M                     %遍历原图像每个像素的灰度值
            for j=1:N
                if (K(i,j)*255>=t)    %分割出灰度值》=t的像素
                    number_A=number_A+1;  %得到A部分总像素
                    hui_A=hui_A+K(i,j);   %得到A部分总灰度值
                elseif (K(i,j)*255<t) %分割出灰度值《t的像素
                    number_B=number_B+1;  %得到B部分总像素
                    hui_B=hui_B+K(i,j);   %得到B部分总灰度值
                end
            end
        end
        PA=number_A/number_all;            %得到A部分像素总数与图像总像素的比列
        PB=number_B/number_all;            %得到B部分像素总数与图像总像素的比列
        A_ave=hui_A*255/number_A;          %得到A部分总灰度值与A部分总像素的比例
        B_ave=hui_B*255/number_B;          %得到B部分总灰度值与B部分总像素的比例
        ICV=PA*((A_ave-all_ave)^2)+PB*((B_ave-all_ave)^2);  %Otsu算法
        scores(t+1)=ICV;
        if (ICV>ICV_t)                     %不断判断，得到最大方差
            ICV_t=ICV;
            k=t;                           %得到最大方差的最优阈值
        end
    end
    
    figure;
    plot(0:255, scores, 'b-', 'LineWidth',2);
    xlabel('Threshold t');
    ylabel("scores")
    title("Score curve vs. Threshold t")
    % k                                      %显示阈值
    I=imbinarize(K,k/255);        %二值化
    % figure,imshow(I),title("Otsu海陆分割方法结果示意");
    imwrite(I, "Otsu海陆分割结果示意/mid_" + fileName);

    %%% 形态学处理
    SE=strel('disk',9);
    SE2=strel('disk',1);
    SE3=strel('disk',1);
    label=imdilate(I,SE);
    label=imdilate(label,SE);
    label=imdilate(label,SE);
    label=imdilate(label,SE);
    % figure,imshow(label);
    label=imerode(label,SE);
    label=imerode(label,SE);
    label=imerode(label,SE);
    label=imfill(label,'holes');
    
    %孤立区域消除
    label=bwareaopen(label,40000);
    % figure,imshow(label);title("形态学处理结果示意");
    imwrite(label, "形态学处理结果示意/Otsu_"+fileName);

    disp(['Elapsed times: ', num2str(toc), ' seconds']);
end
