function harmonization(ref_mean_file, ref_std_file, tar_mean_file, tar_std_file, mask_file, lambda1, output_prefix)
% Harmonization of neuroimaging data using specified reference and target datasets.
%
% Parameters:
% - ref_mean_file: String, path to the NIfTI file of the reference dataset's mean.
% - ref_std_file: String, path to the NIfTI file of the reference dataset's standard deviation.
% - tar_mean_file: String, path to the NIfTI file of the target dataset's mean.
% - tar_std_file: String, path to the NIfTI file of the target dataset's standard deviation.
% - mask_file: String, path to the NIfTI file of the mask to apply to the datasets.
% - lambda1: Double, regularization parameter (suggested 0.1 as optimal in the paper).
% - output_prefix: String, prefix for the output NIfTI files.
%
% This function reads in reference and target mean and standard deviation images,
% applies a mask, and performs harmonization. The harmonized parameters are then
% saved to NIfTI files with the provided output prefix.

% Load the input data
m1ref_info = load_untouch_nii(ref_mean_file);
m1ref = double(m1ref_info.img);
m2ref = double(niftiread(ref_std_file));
m1tar = double(niftiread(tar_mean_file));
m2tar = double(niftiread(tar_std_file));
mask_info = load_untouch_nii(mask_file);
mask = mask_info.img;

% Validate input mask dimensions
if ~isequal(size(mask), size(m1ref)) || ~isequal(size(mask), size(m2ref)) || ~isequal(size(mask), size(m1tar)) || ~isequal(size(mask), size(m2tar))
    error('Mask dimensions must match the dimensions of the input images.');
end

% Set the regularization parameter
if nargin < 6 || isempty(lambda1)
    lambda1 = 0.1; % Default value if not specified
end

% Generate empty arrays for harmonization parameters
aData = zeros(size(mask));
bData = zeros(size(mask));

% Find the harmonization parameters
[aData, bData] = FindParams(m1ref, m2ref, m1tar, m2tar, mask, lambda1);

% Save the output data to NIfTI files
aData_info = m1ref_info; % Preserve header and affine information
aData_info.img = aData;
bData_info = m1ref_info; % Preserve header and affine information
bData_info.img = bData;

save_untouch_nii(aData_info, sprintf('%s_aData.nii.gz', output_prefix));
save_untouch_nii(bData_info, sprintf('%s_bData.nii.gz', output_prefix));

end


function [aData, bData] = FindParams(m1ref, m2ref, m1tar, m2tar, mask, lambda1)
[sx, sy, sz] = size(mask);

aData = zeros(size(mask));
bData = zeros(size(mask));

options = optimset('MaxIter', 400, 'Display', 'off');
x_init = [1, 0];

parfor ix = 1:sx
    for iy = 1:sy
        for iz = 1:sz
            if (mask(ix, iy, iz) >= 0.5)
                lambda2 = (m1ref(ix, iy, iz) / m2ref(ix, iy, iz)) ^ 2; % scale second moment term
                % Check for division by zero or NaN/Inf values
                if isnan(lambda2) || isinf(lambda2)
                    disp(['lambda2 is NaN or Inf at (', num2str(ix), ',', num2str(iy), ',', num2str(iz), ')']);
                    continue;
                end
                
                try
                    x = fminunc(@(x)momentCost(m1ref(ix, iy, iz), m1tar(ix, iy, iz), m2ref(ix, iy, iz), m2tar(ix, iy, iz), x(1), x(2), lambda1, lambda2), x_init, options);
                    aData(ix, iy, iz) = x(1);
                    bData(ix, iy, iz) = x(2);
                catch ME
                    disp(['Error at (', num2str(ix), ',', num2str(iy), ',', num2str(iz), '): ', ME.message]);
                end
            end
        end
    end
end

end


function cost = momentCost(m1ref, m1tar, m2ref, m2tar, a, b, lambda1, lambda2)
FirstMomentDiff = m1ref - (a * m1tar + b);
SecondMomentDiff = m2ref - (a ^ 2 * m2tar);
RegParams = 1 - normpdf(a, 1, 1) / normpdf(1, 1, 1) + 1 - normpdf(b, 0, 1) / normpdf(0, 0, 1);

cost = ( ...
    FirstMomentDiff ^ 2 ...
    + lambda1 .* RegParams ...
    + lambda2 .* SecondMomentDiff ^ 2);
end
