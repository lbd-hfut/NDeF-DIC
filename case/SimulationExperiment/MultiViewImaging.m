% 清除工作区和命令窗口
clear variables
clc
close all

%% 参数定义
% 内参（含畸变）定义
L_pixel = 1440; H_pixel = 1080; % 相机分辨率
pixel_size = 3.45/1000; % 像元尺寸（单位：mm）
f = 4; % 相机像距（单位：mm）
Fx = f/pixel_size; Fy = f/pixel_size; % 等效焦距
Cx = L_pixel/2+0.5; Cy = H_pixel/2+0.5; % 主点坐标
k1 = 0.1; k2 = 0.05; % 畸变系数

% 相机阵列外参定义
Ax_world2refcam = 0; Ay_world2refcam = 0; Az_world2refcam = 0; R_world2refcam = eul2R(Ax_world2refcam,Ay_world2refcam,Az_world2refcam);% 参考相机欧拉角（世界坐标系 to 参考相机坐标系，单位：°）
Tx_world2refcam = 0; Ty_world2refcam = 0; Tz_world2refcam = 500; T_world2refcam = [Tx_world2refcam; Ty_world2refcam; Tz_world2refcam]; % 参考相机平移量（世界坐标系原点在相机坐标系下的坐标）
num_View = 5; % 视角数量
Angle_View = 10; % 相邻相机夹角（单位：°）
k = 1:(num_View-1)/2; % 生成索引
if mod(num_View,2) ~=1
    k = [k k(end)+1];
end

% 第i号相机坐标系到参考相机坐标系的欧拉角
Ax_i2refcam = zeros(1,num_View); 
Ay_i2refcam = [0, reshape([-k; k], 1, [])] * Angle_View/180*pi; % 正负号遵循右手定则
Az_i2refcam = zeros(1,num_View);
% 第i号相机坐标系原点在参考相机坐标系下的坐标
b = '圆弧'; % b取值为“圆弧”和“直线”
if strcmp(b, '圆弧')
    Tx_i2refcam = 2*Tz_world2refcam*sin(Ay_i2refcam/2).*cos(Ay_i2refcam/2);
    Tz_i2refcam = 2*Tz_world2refcam*sin(Ay_i2refcam/2).*sin(Ay_i2refcam/2);
    Ty_i2refcam = zeros(1,num_View);
else
    Tx_i2refcam = zeros(1,num_View);
    for i = 1:num_View
        prompt = ['请输入第', num2str(i), '个数值: '];
        Tx_i2refcam(i) = input(prompt);
    end
    Ty_i2refcam = zeros(1,num_View);
    Tz_i2refcam = zeros(1,num_View);
end


R_refcam2camArr = zeros(3,3,num_View); % 参考相机坐标系到每台相机坐标系的转换关系
T_refcam2camArr = zeros(3,1,num_View);
R_world2camArr = zeros(3,3,num_View); % 世界坐标系到每台相机坐标系的转换关系
T_world2camArr = zeros(3,1,num_View);
for i = 1:num_View
    R_i2ref = eul2R(Ax_i2refcam(i), Ay_i2refcam(i), Az_i2refcam(i));
    T_i2ref = [Tx_i2refcam(i); Ty_i2refcam(i); Tz_i2refcam(i)];
    R_ref2i = R_i2ref^(-1);
    T_ref2i = -R_i2ref^(-1)*T_i2ref;
    R_world2i = R_ref2i*R_world2refcam;
    T_world2i = R_ref2i*T_world2refcam+T_ref2i;
    R_refcam2camArr(:,:,i) = R_ref2i;
    T_refcam2camArr(:,:,i) = T_ref2i;
    R_world2camArr(:,:,i) = R_world2i;
    T_world2camArr(:,:,i) = T_world2i;
end


%% 空间点参数定义
numPoints = 12000000; % 空间点的数量
ImageGray = imread('002.bmp'); % 空间点的灰度/光强信息
rr = 3/5; % 特征点区域占视场比例（单方向比值）
L_image = L_pixel*pixel_size; H_image = H_pixel*pixel_size; % 图像物理尺寸（单位：mm）
L_view = Tz_world2refcam*L_image/f; H_view = Tz_world2refcam*H_image/f; % 视场大小（单位：mm）
Xw_Lower = -L_view/2*rr; Xw_Upper = L_view/2*rr; % 特征点Xw取值范围
Yw_Lower = -H_view/2*rr; Yw_Upper = H_view/2*rr; % 特征点Yw取值范围
a = 'Outer cylindrical surface'; % 取值'Plane'、'Inner cylindrical surface'、'Outer cylindrical surface'、'Sine surface'
% 生成三维散斑场景
[scenePoints, sceneIntensities] = Generating3DScenes(ImageGray, numPoints, Xw_Lower, Xw_Upper, Yw_Lower, Yw_Upper, a);


%% 生成散斑图
noise_std = 2; % 噪声的标准差，灰度
sigma = 0.5; % 高斯模糊的标准差
scenePoints = scenePoints';

for i = 1:num_View
    % 添加噪声
    if i == 1
        rng(25)
    elseif i == 2
        rng(259)
    elseif i == 3
        rng(856)
    elseif i == 4
        rng(125)
    elseif i == 5
        rng(68)
    elseif i == 6
        rng(841)
    elseif i == 7
        rng(489)
    elseif i == 8
        rng(2456)
    elseif i == 9
        rng(269)
    else
        rng(59876)
    end
    
    [uv_distortion, Image_noisy] = GeneratingImage(Fx, Fy, Cx, Cy, R_world2camArr(:,:,i), T_world2camArr(:,:,i), k1, k2, scenePoints, sceneIntensities, noise_std, sigma, L_pixel, H_pixel);
    % 显示散斑图像
    figure;
    imshow(Image_noisy, []);
    title('相机成像图像');
    % 动态生成文件名
    filename = sprintf('Image%d.bmp', i);
    % 保存图像
    imwrite(Image_noisy, filename);
    % 显示保存的文件名
    fprintf('已保存文件：%s\n', filename);
end

fileName = 'CameraArry_Parameters.mat';
save(fileName, 'Fx','Cx','Cy','k1','k2','R_refcam2camArr','T_refcam2camArr','R_world2camArr','T_world2camArr');


%% 函数库

function R = eul2R(Ax, Ay, Az)
Rx = [1 0 0; 0 cos(Ax) sin(Ax); 0 -sin(Ax) cos(Ax)];
Ry = [cos(Ay) 0 -sin(Ay); 0 1 0; sin(Ay) 0 cos(Ay)];
Rz = [cos(Az) sin(Az) 0; -sin(Az) cos(Az) 0; 0 0 1];
R = Rz*Ry*Rx;
end

function [scenePoints, sceneIntensities] = Generating3DScenes(speckleImage, numPoints, x_Lower, x_Upper, y_Lower, y_Upper, a)
    rng(42);
    % 读取散斑图像
    if size(speckleImage, 3) > 1
        speckleImage = rgb2gray(speckleImage); % 转为灰度图
    end
    [imageHeight, imageWidth] = size(speckleImage);

    % 分块计算
    blockSize = 1e6; % 每个块的大小
    numBlocks = ceil(numPoints / blockSize); % 计算块的数量
    sceneIntensities = zeros(numPoints, 1); % 初始化光强数组
    scenePoints = zeros(numPoints, 3); % 初始化散斑点坐标数组
    % 启用并行计算
    if isempty(gcp('nocreate'))
        parpool; % 启动并行池
    end
    % 分块并行计算
    for block = 1:numBlocks
        % 计算当前块的索引范围
        startIdx = (block - 1) * blockSize + 1; % 每个块的起始点序号
        endIdx = min(block * blockSize, numPoints); % 每个块的终止点序号
        numPointsInBlock = endIdx - startIdx + 1; % 每个块的点数量
        % 生成当前块的散斑点 x 和 y 坐标
        x = x_Lower + (x_Upper-x_Lower) * rand(numPointsInBlock, 1); % x 范围为 x_Lower~x_Lower
        y = y_Lower + (y_Upper-y_Lower) * rand(numPointsInBlock, 1); % y 范围为 y_Lower~y_Lower
        % 将 x 和 y 坐标映射到图像像素坐标
        imageX = (x + (x_Upper-x_Lower)/2) / (x_Upper-x_Lower) * (imageWidth - 1) + 1;
        imageY = (y + (y_Upper-y_Lower)/2) / (y_Upper-y_Lower) * (imageHeight - 1) + 1;
        % 使用双线性插值获取光强信息
        blockIntensities = interp2(double(speckleImage), imageX, imageY, 'linear');
        % 根据散斑场景的形状确定 z 坐标
        if strcmp(a, 'Plane')
            z = zeros(size(x)); % 计算 z 坐标
        elseif strcmp(a, 'Inner cylindrical surface')
            radius = x_Upper-x_Lower; % 圆柱面半径；该设定下，圆柱面弧度为60°
            z = sqrt(radius^2 - x.^2)-radius; % 计算 z 坐标
        elseif strcmp(a, 'Outer cylindrical surface')
            radius = x_Upper-x_Lower; % 圆柱面半径；该设定下，圆柱面弧度为60°
            z = -sqrt(radius^2 - x.^2)+radius; % 计算 z 坐标
        else
            radius = x_Upper-x_Lower; % 圆柱面半径；该设定下，圆柱面弧度为60°
            Amplitude = (radius-radius*sin(60/180*pi))/2;  % 正弦曲面幅度
            z = Amplitude * sin(2*pi/(x_Upper-x_Lower) * x);
        end
        % 将当前块的散斑点坐标和光强存储到结果数组中
        scenePoints(startIdx:endIdx, :) = [x, y, z];
        sceneIntensities(startIdx:endIdx) = blockIntensities;
        % 显示进度
        fprintf('Processing block %d/%d...\n', block, numBlocks);
    end
%     % 绘制散斑场景的三维图
%     figure('Position', [100, 100, 500, 400]); % [left, bottom, width, height]
%     Display_step_length = 100;
%     scatter3(scenePoints(1:Display_step_length:end, 1), scenePoints(1:Display_step_length:end, 2), scenePoints(1:Display_step_length:end, 3), 2, sceneIntensities(1:Display_step_length:end), 'filled');
%     colormap gray;
%     colorbar; % 显示颜色条
%     xlabel('X');
%     ylabel('Y');
%     zlabel('Z');
%     zticks(0); % 设置刻度位置
%     set(gca, 'FontSize', 14); % 设置坐标轴字体大小
%     set(gcf, 'DefaultTextFontSize', 14); % 设置图形中所有文本的默认字体大小
%     title(a);
%     axis equal; % 保持坐标轴比例一致
%     view(3); % 设置视角为三维视角
end

function uv_distortion = CoordPixCal_distortion_v3(Fx, Fy, Cx, Cy, R, T, k1, k2, Coord_World)
    numel_points = size(Coord_World,2);
    uv_distortion = zeros(2,numel_points);
    K = [Fx 0 Cx; 0 Fy Cy; 0 0 1];
    for i = 1:numel_points
        % 不含畸变的像素坐标
        uv_Nondistortion_i = K*(R*Coord_World(:,i)+T);
        uv_Nondistortion_i = uv_Nondistortion_i/uv_Nondistortion_i(3);
        % 计算畸变量
        x_Nondistortion_i = (uv_Nondistortion_i(1)-Cx)/Fx;
        y_Nondistortion_i = (uv_Nondistortion_i(2)-Cy)/Fy;
        r_Nondistortion_i = sqrt(x_Nondistortion_i^2+y_Nondistortion_i^2);
        x_distortion_i = x_Nondistortion_i*(1+k1*r_Nondistortion_i^2+k2*r_Nondistortion_i^4);
        y_distortion_i = y_Nondistortion_i*(1+k1*r_Nondistortion_i^2+k2*r_Nondistortion_i^4);
        u_distortion_i = Fx*x_distortion_i+Cx;
        v_distortion_i = Fy*y_distortion_i+Cy;
        % 生成含畸变的像素坐标
        uv_distortion(1,i) = u_distortion_i;
        uv_distortion(2,i) = v_distortion_i;
    end
end

function [uv_distortion, speckleImage_noisy] = GeneratingImage(Fx, Fy, Cx, Cy, R, T, k1, k2, Coord_World, sceneIntensities, noise_std, sigma, L_pixel, H_pixel)
uv_distortion = CoordPixCal_distortion_v3(Fx, Fy, Cx, Cy, R, T, k1, k2, Coord_World);
u = uv_distortion(1,:);
v = uv_distortion(2,:);

% 过滤超出图像范围的点
validIndices = u >= 1 & u <= L_pixel & v >= 1 & v <= H_pixel;
u = u(validIndices);
v = v(validIndices);
intensities = sceneIntensities(validIndices);
% 生成散斑图像
intensitySum = zeros(H_pixel, L_pixel); % 用于记录每个像素点的光强总和
count = zeros(H_pixel, L_pixel); % 用于记录每个像素点被访问的次数
for ii = 1:length(u)
    row = round(v(ii));
    col = round(u(ii));
    if row >= 1 && row <= H_pixel && col >= 1 && col <= L_pixel
        intensitySum(row, col) = intensitySum(row, col) + intensities(ii); % 累加光强
        count(row, col) = count(row, col) + 1; % 增加访问次数
    end
end
% 计算每个像素点的平均光强
speckleImage = intensitySum ./ count;
speckleImage(isnan(speckleImage)) = 0; % 将 NaN 值替换为 0
% 线性缩放
speckleImage_0 = speckleImage - min(speckleImage(:)); % 将最小值映射到 0
speckleImage = speckleImage_0 / max(speckleImage_0(:)) * 240; % 将最大值映射到 240
speckleImage = uint8(round(speckleImage(:,:)));
% 对散斑图案进行高斯模糊，模拟相机成像
speckleImage = imgaussfilt(speckleImage, sigma);

speckleImage_noisy(:,:) = addNoiseToPixels(speckleImage, noise_std);

end

function speckleImage_noisy = addNoiseToPixels(speckleImage, noise_std)
    % 添加随机噪声到像素坐标
    % uv_pixels: 原始像素坐标 (2xN)
    % noise_std: 噪声的标准差（幅值）

    % 噪声的标准差
    std_dev = noise_std;

    % 噪声的均值（零均值）
%     mean_val = 0;

    % 生成随机噪声
    noise = std_dev * randn(size(speckleImage)); % 噪声矩阵

    % 添加噪声到原始像素坐标
    speckleImage_noisy = speckleImage + uint8(noise);
end
