load PCA_data

epochs = 10;
v_num = 60;
s_length = 806;
mean_v = zeros(1,882);
u = zeros(v_num,882);
v = zeros(v_num,882);

tt = 1;
for i = 1:epochs
    for j = 1:s_length
        sa = daa(j,:);
        mean_v = ((tt-1)*mean_v+sa)/tt;
        ssa = sa - mean_v;
        u(1,:) = ssa;
        if i == 1 && j >= 2 && j <= v_num+1
            v(j-1,:) = ssa;
        else
           t = tt - v_num;
           u(1,:) = ssa;
           for k =1:v_num 
                y = u(k,:)*(v(k,:)'/norm(v(k,:),2));
               v(k,:) = ((t-1)*v(k,:)+u(k,:)*y)/t;
               u(k+1,:) = u(k,:) - u(k,:)*(v(k,:)'/norm(v(k,:),2))*(v(k,:)/norm(v(k,:),2));
           end    
        end
        tt = tt+1;
    end    
end

for i = 1:v_num
    v(i,:) = v(i,:)/norm(v(i,:),2);
end

meanvalue = mean_v;
Uk = v';
db = da*Uk;
maxV = max(db(:));
minV = min(db(:));

dc = bsxfun(@minus, db, minV);
dc = round(dc./(maxV-minV)*127);

save('cciPCA.mat','Uk','meanvalue','da','dc','maxV','minV');