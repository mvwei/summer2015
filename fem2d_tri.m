function [] = fem2d()

global g_w
global g_locs

g_w = [1/6,1/6,1/6];
g_locs = [1/6,1/6; 2/3, 1/6; 1/6, 2/3];

[shps,pts,pts_ind,edge_pts]=mkgridsq(10,10, pi(), pi());

A = zeros(length(pts_ind),length(pts_ind));
f = zeros(length(pts_ind),1);
source = ones(length(pts_ind),1);
real = zeros(length(pts_ind),1);

for i = 1:length(source)
    x = pts_ind(i, 2);
    y = pts_ind(i, 3);
    source(i) = -8*sin(2*x)*sin(2*y);
    real(i) = sin(2*x)*sin(2*y);
end

pts_local = zeros(length(shps(1,:)),3);

for i = 1:length(shps)
    %for every element, calculate the integrals, add it to the larger
    %matrix.
    
    if mod(i,10)==1
        i
    end

    for j = 1:length(pts_local)  
        pts_local(j,:) = pts_ind(shps(i,j),:);
    end

    A = A + calcA(pts_local, length(A));
    f = f + source.*calcF(pts_local, length(A));
end

%Boundary conditions here.
for i = 1:length(edge_pts)
   x = edge_pts(i);
   A(x,:) = 0;
   A(:,x) = 0;
   A(x,x) = 1;
   f(x) = 0; 
end

sol = A\f;
max(sol)
%sol = zeros(length(pts_ind), 1);

%scatter3(pts_ind(:,2),pts_ind(:,3),sol)

hold on

for i = 1:length(shps)
   tmp = shps(i,:);
   x = zeros(1,length(tmp)+1);
   y = zeros(1,length(tmp)+1);
   z1 = zeros(1,length(tmp)+1);
   z2 = zeros(1,length(tmp)+1);
   for j = 1:length(tmp)
      x(j) = pts_ind(tmp(j),2);
      y(j) = pts_ind(tmp(j),3);
      z1(j) = sol(tmp(j));
      z2(j) = real(tmp(j));
   end
   x(length(tmp)+1) = x(1);
   y(length(tmp)+1) = y(1);
   z1(length(tmp)+1) = z1(1);
   z2(length(tmp)+1) = z2(1);
   line(x,y,z1)
   line(x,y,z2,'Color','red')
end

sum((real - sol).^2)

size(real)
size(sol)

end

% INPUT
% pts (n x 3 matrix, where n is the number of vertices. First column
% is the point index, second and third columns are (x, y) coordinates of point.

% globn (int): number of total points in the system.

% RETURNS: the value of integral(grad(phi_i) . grad(phi_j) dxdy), where
% phi_i and phi_j are the numbered basis functions of the rectangle,
% where phi_i(i) = d(i,j) and d is the kronecker delta and i, j are indices
% of vertices.

function mat = calcA(pts, globn)

global dpdz
global dpde
global g_w
global g_locs

mat = zeros(globn, globn);

% Shape function for right angle triangle. z from [-1, 1], e from [-1, 1].
% p1 = 1-e-z
% p2 = e
% p3 = z

dp1de = @(z,e) -1;
dp2de = @(z,e) 1;
dp3de = @(z,e) 0;

dp1dz = @(z,e) -1;
dp2dz = @(z,e) 0;
dp3dz = @(z,e) 1;

dpdz = {dp1dz, dp2dz, dp3dz};
dpde = {dp1de, dp2de, dp3de};

%local basis functions, derivatives of all flavors.
for i = 1:length(pts)
    for j = i:length(pts)
        
        int_val = 0;
        %here we take the grad(phi_i)*grad(phi_j), using three-point
        %Gaussian quadrature.
        
        for l = 1:length(g_locs)
                x = g_locs(l,1);
                y = g_locs(l,2);
                jac_tmp = jac(pts,x,y);
                temp = (dpdx(pts,x,y,i) * dpdx(pts, x,y,j) +...
                    dpdy(pts,x,y,i) * dpdy(pts,x,y,j)) / jac_tmp^2;
                
                int_val = int_val + g_w(l)*temp*jac_tmp;
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

p1 = @(z,e) 1-e-z;
p2 = @(z,e) e;
p3 = @(z,e) z;

p = {p1, p2, p3};

for i = 1:length(pts)
   sum = 0;
    for l = 1:length(g_w)
            x = g_locs(l,1);
            y = g_locs(l,2);
            sum = sum + g_w(l)*jac(pts,x,y)*p{i}(x,y);
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

% lind is local index, that is: which basis is used.
% RETURNS VALUE THAT HAS NOT BEEN DIVIDED BY JACOBIAN. (Because I don't
% want to call the Jacobian that many times.)
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


% Create grid, squares.
% FOR TRIANGLES!
function [shps, pts, pts_ind, nepoints] = mkgridsq(nc, nr, lenr, lenc)

intr = lenr/nr;
intc = lenc/nc;

numshapes = nr*nc*2;
numpoints = (nr+1)*(nc+1);
scount = 1;
shps = zeros(numshapes, 3);
pts = zeros(nr+1,nc+1);
pts_ind = zeros(numpoints, 3);

nepoints = zeros(1,(nr+1)*2 + (nc-1)*2); % number of edge points for a rectangular domain only.

p1 = 0;
p2 = 0;
p3 = 0;

epc = 1; %edge point counter

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
                
            p1 = pts(i,j);
            p2 = p1 + (nc+1);
            p3 = p1 + 1;
            
            t2 = p1 + 1;
            t1 = t2 + nc + 1;
            t3 = t1 - 1;

            shps(scount, :) = [p1,p2,p3];
            shps(scount + 1, :) = [t1,t2,t3];
            scount = scount + 2;
            
        end
    end

end

