close all;clear ;clc;
% ��ȡ��ǰ�ļ���������PNG�ļ�
training_path = "YOUR_TRAINING_IMAGES_PATH"
pictureFiles = dir('YOUR_TRAINING_IMAGES_PATH/*.png')

test_path = "YOUR_TEST_IMAGES_PATH"
test_pictureFiles = dir('YOUR_TEST_IMAGES_PATH/*.png')
%training_path = test_path
%pictureFiles = test_pictureFiles



% ��������PNG�ļ�
for k=1: length(pictureFiles)
    tic;
    fileName = pictureFiles(k).name;
    IN=im2double(imread(training_path + fileName));  %��Ϊ˫���ȣ���0-1

    figure;
    imhist(imread(training_path + fileName));  %axis([0 1,0,600000]);      %��ʾ�Ҷ�ֱ��ͼ

    IN = IN(:,:,1);
    J=wiener2(IN);  %����Ӧ�˲�
    K=medfilt2(J);  %��ֵ�˲�

    [M,N]=size(IN);                     %�õ�ͼ����������
    number_all=M*N;                    %������ֵ
    hui_all=0;                         %Ԥ��ͼ���ܻҶ�ֵΪ0
    ICV_t=0;                           %Ԥ����󷽲�Ϊ0
    %�õ�ͼ���ܻҶ�ֵ
    for i=1:M
        for j=1:N
            hui_all=hui_all+K(i,j);
        end
    end
    all_ave=hui_all*255/number_all;   %ͼ��Ҷ�ֵ����ƽ��ֵ
     
     
    %tΪĳ����ֵ����ԭͼ���ΪA���֣�ÿ������ֵ>=t����B���֣�ÿ������ֵ<t��
    scores = zeros(1,256);
    for t=0:255                       %������̽����tֵ
        hui_A=0;                      %��������A�����ܻҶ�ֵ
        hui_B=0;                      %��������B�����ܻҶ�ֵ
        number_A=0;                   %��������A����������
        number_B=0;                   %��������B����������
        for i=1:M                     %����ԭͼ��ÿ�����صĻҶ�ֵ
            for j=1:N
                if (K(i,j)*255>=t)    %�ָ���Ҷ�ֵ��=t������
                    number_A=number_A+1;  %�õ�A����������
                    hui_A=hui_A+K(i,j);   %�õ�A�����ܻҶ�ֵ
                elseif (K(i,j)*255<t) %�ָ���Ҷ�ֵ��t������
                    number_B=number_B+1;  %�õ�B����������
                    hui_B=hui_B+K(i,j);   %�õ�B�����ܻҶ�ֵ
                end
            end
        end
        PA=number_A/number_all;            %�õ�A��������������ͼ�������صı���
        PB=number_B/number_all;            %�õ�B��������������ͼ�������صı���
        A_ave=hui_A*255/number_A;          %�õ�A�����ܻҶ�ֵ��A���������صı���
        B_ave=hui_B*255/number_B;          %�õ�B�����ܻҶ�ֵ��B���������صı���
        ICV=PA*((A_ave-all_ave)^2)+PB*((B_ave-all_ave)^2);  %Otsu�㷨
        scores(t+1)=ICV;
        if (ICV>ICV_t)                     %�����жϣ��õ���󷽲�
            ICV_t=ICV;
            k=t;                           %�õ���󷽲��������ֵ
        end
    end
    
    figure;
    plot(0:255, scores, 'b-', 'LineWidth',2);
    xlabel('Threshold t');
    ylabel("scores")
    title("Score curve vs. Threshold t")
    % k                                      %��ʾ��ֵ
    I=imbinarize(K,k/255);        %��ֵ��
    % figure,imshow(I),title("Otsu��½�ָ�����ʾ��");
    imwrite(I, "Otsu��½�ָ���ʾ��/mid_" + fileName);

    %%% ��̬ѧ����
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
    
    %������������
    label=bwareaopen(label,40000);
    % figure,imshow(label);title("��̬ѧ������ʾ��");
    imwrite(label, "��̬ѧ������ʾ��/Otsu_"+fileName);

    disp(['Elapsed times: ', num2str(toc), ' seconds']);
end
