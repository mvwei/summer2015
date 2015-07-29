function [] = fem2d()

global g_w
global g_locs

% Declare Gaussian quadrature indices and weights. Using n=3 quadrature
% here.
g_w = [5/9, 8/9, 5/9];
g_locs = [-sqrt(3/5),0,sqrt(3/5)];

% shps: nshapes x nvertices matrix; at index i, shps(i,:) contains the
% global vertex indices.
% pts_ind: npoints_global x 3, the global vertex indices and the locations
% of the points.
% edge_pts: the indices of points that are on the boundary's edge.
[shps,pts,pts_ind,edge_pts]=mkgridsq(20,20, 2,2)

A = zeros(length(pts_ind),length(pts_ind));
f = zeros(length(pts_ind),1);
source = ones(length(pts_ind),1);

pts_local = zeros(length(shps(1,:)),3);

for i = 1:length(shps)
    %for every element, calculate the integrals, add it to the larger
    %matrix.
    if mod(i,10) == 1
       i 
    end
    
    for j = 1:length(pts_local)  
        pts_local(j,:) = pts_ind(shps(i,j),:);
    end

    A = A + calcA(pts_local, length(A));
    f = f + source.*calcF(pts_local, length(A));
end

% Apply Dirichlet boundary conditions here.
for i = 1:length(edge_pts)
   x = edge_pts(i);
   A(x,:) = 0;
   A(:,x) = 0;
   A(x,x) = 1;
   f(x) = 0; 
end

sol = A\f;
max(sol)

hold on
scatter3(pts_ind(:,2),pts_ind(:,3),sol)

for i = 1:length(shps)
   tmp = shps(i,:);
   x = zeros(1,length(tmp));
   y = zeros(1,length(tmp));
   z = zeros(1,length(tmp));
   for j = 1:length(tmp)
      x(j) = pts_ind(tmp(j),2);
      y(j) = pts_ind(tmp(j),3);
      z(j) = sol(tmp(j));
   end
   line(x,y,z)
end

end

% calcA: calculates the individual shape's contribution to the "A" matrix.

% INPUT
% pts (n x 3 matrix, where n is the number of vertices PER SHAPE. First column
% is the point index, second and third columns are (x, y) coordinates of point.

% globn (int): number of total points in the system.

% RETURNS: 
% mat, globn x globn matrix.
% the value of integral(grad(phi_i) . grad(phi_j) dxdy), where
% phi_i and phi_j are the numbered basis functions of the rectangle,
% where phi_i(i) = d(i,j) and d is the kronecker delta and i, j are indices
% of vertices.

function mat = calcA(pts, globn)

global dpdz
global dpde
global g_w
global g_locs

mat = zeros(globn, globn);

% Shape functions for z from [-1,1] and e from [-1,1]
% p1 = (z-1)*(e-1)/4
% p2 = -(z+1)*(e-1)/4
% p3 = (z+1)*(e+1)/4
% p4 = -(z-1)*(e+1)/4

%pre-calculated derivatives of the master shape, for ease. 

dp1de = @(z,e) (z-1)/4;
dp2de = @(z,e) (-1-z)/4;
dp3de = @(z,e) (z+1)/4;
dp4de = @(z,e) (1-z)/4;

dp1dz = @(z,e) (e-1)/4;
dp2dz = @(z,e) (1-e)/4;
dp3dz = @(z,e) (e+1)/4;
dp4dz = @(z,e) (-e-1)/4;

% cell array of derivatives. Easiest way to do the x_i * phi_i.
dpdz = {dp1dz, dp2dz, dp3dz, dp4dz};
dpde = {dp1de, dp2de, dp3de, dp4de};

for i = 1:length(pts)
    for j = i:length(pts)
        
        int_val = 0;
        % here we take the grad(phi_i)*grad(phi_j), using Gaussian
        % quadrature specified above.
        
        for l = 1:length(g_locs)
            for k = 1:length(g_locs)
                x = g_locs(l);
                y = g_locs(k);
                jac_temp = jac(pts, x, y);
                
                temp = (dpdx(pts,x,y,i) * dpdx(pts, x,y,j) +...
                    dpdy(pts,x,y,i) * dpdy(pts,x,y,j)) / jac_temp^2;
                
                int_val = int_val + g_w(l)*g_w(k)*temp*jac_temp;
            end
        end
        
        mat(pts(i,1),pts(j,1)) = int_val;
        mat(pts(j,1),pts(i,1)) = int_val;
    end
end

end

function f = calcF(pts, globn)

global g_w
global g_locs

f = zeros(globn,1);

% Shape functions for z from [-1,1] and e from [-1,1]
p1 = @(z,e) (z-1)*(e-1)/4;
p2 = @(z,e) -(z+1)*(e-1)/4;
p3 = @(z,e) (z+1)*(e+1)/4;
p4 = @(z,e) -(z-1)*(e+1)/4;

p = {p1, p2, p3, p4};

for i = 1:length(pts)
   sum = 0;
    for l = 1:length(g_w)
        for k = 1:length(g_w)
            x = g_locs(l);
            y = g_locs(k);
            sum = sum + g_w(l)*g_w(k)*jac(pts,x,y)*p{i}(x,y);
        end
    end
    
    f(pts(i,1)) = sum;
end

end

function j = jac(pts, z, e)

global dpdz
global dpde

dxdz = 0;
dxde = 0;
dydz = 0;
dyde = 0;

for i = 1:length(pts)
   dxdz = dxdz + pts(i,2)*dpdz{i}(z,e);
   dxde = dxde + pts(i,2)*dpde{i}(z,e);
   dydz = dydz + pts(i,3)*dpdz{i}(z,e);
   dyde = dyde + pts(i,3)*dpde{i}(z,e); 
end

j = dxdz*dyde - dxde*dydz;

end

% FUNCTIONS: DPDX DPDY
% INPUT: pts (4x3 int, double, double), z (double), e (double), ind (int)
% OUTPUT: x (double)

% Returns the value of dpdx for a certain basis function (phi_ind).
% RETURNS VALUE OF DPDX THAT HAS NOT BEEN DIVIDED BY JACOBIAN.

function x = dpdx(pts, z, e, ind)
   
    global dpdz
    global dpde
    
    p1 = 0; % for dpdx
    p2 = 0;
    
    for i = 1:length(pts)
        p1 = p1 + pts(i,3)*dpde{i}(z,e);
        p2 = p2 + pts(i,3)*dpdz{i}(z,e);
    end
    
    x = dpdz{ind}(z,e)*p1 - dpde{ind}(z,e)*p2;

end

function x = dpdy(pts, z, e, ind)
    
    global dpdz
    global dpde
    
    p1 = 0;
    p2 = 0;
    
    for i = 1:length(pts)
        p1 = p1 + pts(i,2)*dpde{i}(z,e);
        p2 = p2 + pts(i,2)*dpdz{i}(z,e);
    end
    
    x = -dpdz{ind}(z,e)*p1 + dpde{ind}(z,e)*p2;

end


% FUNCTION mkgridsq:

% INPUT: nc (int), nr (int), lenr (double), lenc (double).
% OUTPUT: shps ( (nc x nr) x 4 ints), pts ( nc + 1 x nr + 1 ints )
% pts_ind ( (nc + 1 x nr + 1) x 3 doubles ), nepoints ( 1 x edgepoints ). 


function [shps, pts, pts_ind, nepoints] = mkgridsq(nc, nr, lenr, lenc)

intr = lenr/nr;
intc = lenc/nc;

numshapes = nr*nc;
numpoints = (nr+1)*(nc+1);
scount = 1;
shps = zeros(numshapes, 4);
pts = zeros(nr+1,nc+1);
pts_ind = zeros(numpoints, 3);

nepoints = zeros(1,(nr+1)*2 + (nc-1)*2); % number of edge points for a square only.

BL = 0;
BR = 0;
TR = 0;
TL = 0;

epc = 1; %edge point counter

% this implementation is for the very simple 

% 11 12 13 14 15 ...
% 6  7  8  9  10
% 1  2  3  4  5

% numbering system. Undoubtedly there are more efficient ways to do this.
% Can be changed later.

    for i = 1:nr+1
        for j = 1:nc+1
            
            pts(i,j) = (i-1)*(nc+1)+j;
            pts_ind(pts(i,j), 1) = pts(i,j);
            pts_ind(pts(i,j), 2) = (j-1)*intc;
            pts_ind(pts(i,j), 3) = (i-1)*intr;
            
            if i==1 || j == 1 || i == nr+1 || j == nc+1
                nepoints(epc) = pts(i,j);
                epc = epc + 1;
            end
            
        end
    end
    
    for i = 1:nr
        for j = 1:nc
            
            BL = pts(i,j);
            BR = BL + 1;
            TR = BR + (nc+1);
            TL = BL + (nc+1);
            shps(scount, :) = [BL, BR, TR, TL];
            scount = scount + 1;
            
        end
    end  
end

