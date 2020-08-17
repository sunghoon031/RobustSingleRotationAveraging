clc;

%% Example: Average 100 rotations (50 inliers, 50 outliers)

n_inliers = 50;
n_outliers = 50;
inlier_noise_level = 5; %deg;
R_true = RandomRotation(pi); 


% 1. Create input rotaions:
n_samples = n_inliers + n_outliers;
R_samples = cell(1, n_samples);
                
for i = 1:n_samples
    if (i <= n_outliers)
        % Inliers: perturb by 5 deg.
        axis_perturb = rand(3,1)-0.5;
        axis_perturb = axis_perturb/norm(axis_perturb);
        angle_perturb = normrnd(0,inlier_noise_level/180*pi); 
        R_perturb = RotationFromUnitAxisAngle(axis_perturb, angle_perturb);
        R_samples{i} = R_perturb*R_true;
    else
        % Outliers: completely random.
        R_samples{i} = RandomRotation(pi); 
    end   
end

% 2-a. Average them using Hartley's L1 geodesic method 
% (with our initialization and outlier rejection scheme):

b_outlier_rejection = true;
n_iterations = 10;
thr_convergence = 0.001;
tic;
R_geodesic = GeodesicL1Mean(R_samples, b_outlier_rejection, n_iterations, thr_convergence);
time_geodesic = toc;

% 2-b. Average them using our approximate L1 chordal method 
% (with our initialization and outlier rejection shceme)

b_outlier_rejection = true;
n_iterations = 10;
thr_convergence = 0.001;
tic;
R_chordal = ChordalL1Mean(R_samples, b_outlier_rejection, n_iterations, thr_convergence);
time_chordal = toc;


% 3. Evaluate the rotation error (deg):

error_GeodesicL1Mean = abs(acosd((trace(R_true*R_geodesic')-1)/2));
error_ChordalL1Mean = abs(acosd((trace(R_true*R_chordal')-1)/2));

disp(['Error (geodesic L1 mean) = ', num2str(error_GeodesicL1Mean), ' deg, took ', num2str(time_geodesic*1000), ' ms'])
disp(['Error (chordal L1 mean) = ', num2str(error_ChordalL1Mean), ' deg, took ', num2str(time_chordal*1000), ' ms' ])
disp('')



%% Function definitions

function out = logarithm_map(in)
    cos_theta = (trace(in)-1)/2;
    sin_theta = sqrt(1-cos_theta^2);
    theta = acos(cos_theta);
    ln_R = theta/(2*sin_theta)*(in-in');
    out = [ln_R(3,2);ln_R(1,3);ln_R(2,1)];
end


function out = SkewSymmetricMatrix(in)
    out=[0 -in(3) in(2) ; in(3) 0 -in(1) ; -in(2) in(1) 0 ];
end

function R = RandomRotation(max_angle_rad)

    unit_axis = rand(3,1)-0.5;
    unit_axis = unit_axis/norm(unit_axis);
    angle = rand*max_angle_rad;
    R = RotationFromUnitAxisAngle(unit_axis, angle);

end

function R = RotationFromUnitAxisAngle(unit_axis, angle)
    
    if (angle==0)
        R = eye(3);
    else
        so3 = SkewSymmetricMatrix(unit_axis);
        R = eye(3)+so3*sin(angle)+so3^2*(1-cos(angle));
    end
end

function R = ProjectOntoSO3(M)   
    [U,~,V] = svd(M);
    R = U*V.';
    if (det(R) < 0)
        V(:,3) = -V(:,3);
        R = U*V.';
    end
end


function R = GeodesicL1Mean(R_input, b_outlier_rejection, n_iterations, thr_convergence)
    
    % 1. Initialize
    
    n_samples = length(R_input);
    
    vectors_total = zeros(9,n_samples);
    for i = 1:n_samples
        vectors_total(:,i)= R_input{i}(:);
    end
    s = median(vectors_total,2);
    
    [U,~,V] = svd(reshape(s, [3 3]));
    R = U*V.';
    if (det(R) < 0)
        V(:,3) = -V(:,3);
        R = U*V.';
    end
    
    % 2. Optimize
    
    for j = 1:n_iterations

        vs = zeros(3,n_samples);
        v_norms = zeros(1,n_samples);
        for i = 1:n_samples
            v =  logarithm_map(R_input{i}*R');
            v_norm = norm(v);
            vs(:,i) = v;
            v_norms(i) = v_norm;
        end
        
        % Compute the inlier threshold (if we reject outliers).
        thr = inf;
        if (b_outlier_rejection)
            sorted_v_norms = sort(v_norms);
            v_norm_firstQ = sorted_v_norms(ceil(n_samples/4));
            if (n_samples <= 50)
                thr = max(v_norm_firstQ, 1);

            else
                thr = max(v_norm_firstQ, 0.5);
            end
        end

        step_num = 0;
        step_den = 0;

        for i = 1:n_samples
            v =  vs(:,i);
            v_norm = v_norms(i);
            if (v_norm > thr)
                continue;
            end
            step_num = step_num + v/v_norm;
            step_den = step_den + 1/v_norm;
        end

        delta = step_num/step_den;
        delta_angle = norm(delta);
        delta_axis = delta/delta_angle;
        
        R_delta = RotationFromUnitAxisAngle(delta_axis, delta_angle);
        R = R_delta*R;
        if (delta_angle < thr_convergence)
            break;
        end
    end
end


function R = ChordalL1Mean(R_input, b_outlier_rejection, n_iterations, thr_convergence)
    
    % 1. Initialize
    n_samples = length(R_input);
    
    vectors_total = zeros(9,n_samples);
    for i = 1:n_samples
        vectors_total(:,i)= R_input{i}(:);
    end      

    s = median(vectors_total,2);
                
    % 2. Optimize
    for j = 1:n_iterations
        if (sum(sum(abs(vectors_total-s))==0) ~= 0)
            s = s+rand(size(s,1),1)*0.001;
        end

        v_norms = zeros(1,n_samples);
        for i = 1:n_samples
            v =  vectors_total(:,i)-s;
            v_norm = norm(v);
            v_norms(i) = v_norm;
        end

        % Compute the inlier threshold (if we reject outliers).
        thr = inf;
        if (b_outlier_rejection)
            sorted_v_norms = sort(v_norms);
            v_norm_firstQ = sorted_v_norms(ceil(n_samples/4));

            if (n_samples <= 50)
                thr = max(v_norm_firstQ, 1.356);
                % 2*sqrt(2)*sin(1/2) is approximately 1.356
            else
                thr = max(v_norm_firstQ, 0.7);
                % 2*sqrt(2)*sin(0.5/2) is approximately 0.7
            end
        end

        step_num = 0;
        step_den = 0;

        for i = 1:n_samples
            v_norm = v_norms(i);
            if (v_norm > thr)
                continue;
            end
            step_num = step_num + vectors_total(:,i)/v_norm;
            step_den = step_den + 1/v_norm;
        end


        s_prev = s;
        s = step_num/step_den;

        update_medvec = s-s_prev;
        if (norm(update_medvec) < thr_convergence)
            break;
        end

    end
    
    R = ProjectOntoSO3(reshape(s, [3 3]));

end