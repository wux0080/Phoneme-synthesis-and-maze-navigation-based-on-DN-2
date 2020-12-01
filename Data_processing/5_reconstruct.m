load cciPCA
load actions.txt
load PCA_data

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


a = zeros(806,60);
for i=1:806
    a(i,:) = actions(i*2,1:60)-1;
end 

ts = zeros(806,882);
for j=1:806
    ta = a(j,:)./127.*(maxV-minV);
    ta = bsxfun(@plus, ta, minV);
    temp = ta*Uk'+meanvalue;
    ts(j,:) = temp;
end

ins = 20;
os =zeros(1,(frames(1,ins)+1)*441);
gs =zeros(1,(frames(1,ins)+1)*441);

os(1,1:441) = daa(upf(1,ins)+1,1:441);
gs(1,1:441) = ts(upf(1,ins)+1,1:441);
for j = 2:frames(1,ins)
    os(1,(j-1)*441+1:(j-1)*441+441) = 0.5*(daa(upf(1,ins)+j-1,442:882)+daa(upf(1,ins)+j,1:441));
    gs(1,(j-1)*441+1:(j-1)*441+441) = 0.5*(ts(upf(1,ins)+j-1,442:882)+ts(upf(1,ins)+j,1:441));
end   

os(1,frames(1,ins)*441+1:(frames(1,ins)+1)*441) = daa(upf(1,ins)+frames(1,ins),442:882);
gs(1,frames(1,ins)*441+1:(frames(1,ins)+1)*441) = ts(upf(1,ins)+frames(1,ins),442:882);

% % index = 44;
% % 
% % ds = daa(index,:);
% % 
% % plot(ds,'linewidth',2);
% %  
% % hold on;
% %  
% % plot(ts(index,:),'linewidth',2);
% set(gca,'xtick',[],'ytick',[]);

% % plot(os,'linewidth',1);

% % hold on;
% % plot(gs,'linewidth',1);
ns = num2str(ins);
filename = ['generated_phonemes/', ns, '_g.wav'];
f = 44100;
audiowrite(filename, gs,f);