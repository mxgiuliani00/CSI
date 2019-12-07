function [R2] = rsq(input1,input2)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
% Input1=dati osservati
% Input2=dati da modello

RSS_=(input1-input2).^2;
RSS1=sum(RSS_);
r_avg=mean(input1);
TSS1_=(input1-r_avg).^2;
TSS1=sum(TSS1_);
R2=1-RSS1/TSS1;

end

