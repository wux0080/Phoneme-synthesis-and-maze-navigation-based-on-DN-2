f=[20,40,80,160,320,640,1280,2560,5120,10240,20480];

h=zeros(11,10,882); %%1764-40ms
ts=2*pi/10;
for m=1:11
    for n=1:10
        for k=1:882
           h(m,n,k)=sin(2*pi*f(m)*(k/44100)-ts*(n-1));
        end
    end
end
save sine_matrix_20ms h
