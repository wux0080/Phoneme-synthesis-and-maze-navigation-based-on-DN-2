load sine_matrix_20ms
load samples
%%2 levels of volume vectors
v1 = squeeze(v(1,:,:));
v2 = squeeze(v(2,:,:));

train_path = 'data/train/';
%test_path = 'data/test/';

num = 20;
frames = zeros(1,20);
frames(1,1) = 20;
frames(1,2) = 25;
frames(1,3) = 28;
frames(1,4) = 33;
frames(1,5) = 35;
frames(1,6) = 40;
frames(1,7) = 41;
frames(1,8) = 66;
frames(1,9) = 74;
frames(1,10) = 70;
frames(1,11) = 45;
frames(1,12) = 34;
frames(1,13) = 44;
frames(1,14) = 54;
frames(1,15) = 62;
frames(1,16) = 20;
frames(1,17) = 27;
frames(1,18) = 20;
frames(1,19) = 39;
frames(1,20) = 29;

total = frames(1,1);
upf = zeros(1,20);
upf(1,1) = 0;
for i =2:20
    upf(1,i) = upf(1,i-1)+frames(1,i-1);
    total = total+frames(1,i);
end

maxlength = 74;

data = zeros(num,maxlength,882);  %%1764-40ms
da = zeros(total+84,882);
daa = zeros(total,882);
ds = zeros(num+1,maxlength+4,11,10);
input = cell(1,total+84);

%%%add silence frames  50ms
silences = zeros(num+1,4,882);
[sd,sfs] = audioread('data/silence.wav');
sd1 = sd(:,1);
for i = 1:(num+1)
    startid = unidrnd(190);
    for j = 1:4
      silences(i,j,:) = sd1((startid+j-1)*441+1:(startid+j-1)*441+882);
    end
end

for i = 1:num
    if i<10
        name = ['0',num2str(i)];
    else
        name = num2str(i);
    end
    train_filename = [train_path,name,'.wav'];   %%change

    [d,fs] = audioread(train_filename);
    d1 = d(:,1);   
    
    for jj = 1:4
        da(upf(1,i)+4*(i-1)+jj,:) = silences(i,jj,:);
    end    
    for j=1:frames(1,i)
        data(i,j,:) = d1(1+(j-1)*441:882+(j-1)*441);
        da(upf(1,i)+4*i+j,:) = d1(1+(j-1)*441:882+(j-1)*441);
        daa(upf(1,i)+j,:) = d1(1+(j-1)*441:882+(j-1)*441);
    end    

    for jj = 1:4
        for m=1:11
            for n=1:10
                for k=1:882
                    ds(i,jj,m,n)=ds(i,jj,m,n)+silences(i,jj,k)*h(m,n,k); 
                end
            end
        end 
        input{1,upf(1,i)+(i-1)*4+jj}.x1 = reshape(ds(i,jj,:,:),11,10);
        input{1,upf(1,i)+(i-1)*4+jj}.x2 = v1;
    end
    for j = 1:frames(1,i)
        for m=1:11
            for n=1:10
                for k=1:882
                    ds(i,j,m,n)=ds(i,j,m,n)+data(i,j,k)*h(m,n,k); 
                end
            end
        end 
        input{1,upf(1,i)+i*4+j}.x1 = reshape(ds(i,j,:,:),11,10);
        input{1,upf(1,i)+i*4+j}.x2 = v2;
    end
end

for jj = 1:4
    da(upf(1,20)+frames(1,20)+4*20+jj,:) = silences(21,jj,:);
end 

for jj = 1:4
    for m=1:11
        for n=1:10
            for k=1:882
                ds(21,jj,m,n)=ds(21,jj,m,n)+silences(i,jj,k)*h(m,n,k); 
            end
        end
    end 
    input{1,upf(1,20)+frames(1,20)+20*4+jj}.x1 = reshape(ds(21,jj,:,:),11,10);
    input{1,upf(1,20)+frames(1,20)+20*4+jj}.x2 = v1;
end


save('PCA_data.mat','da','daa');
save('input_data.mat','input');