function vmiplotraces(vmistats)

traces=vmistats.traces;
delaysN=length(vmistats.delays);
pname=vmistats.subfolder;

figure; hold on; h=[];inonempty=[];

for i=1:6, if ~isempty(traces{i}), 
            inonempty=[inonempty i];
            %h(i)=plot(traces{i}(:,1)/delaysN,traces{i}(:,2),'.-');
            h(i)=plot(traces{i}(:,3),traces{i}(:,2),'.-');
     %       [B,IX]=sort(traces{i}(:,1));
     %       h(i)=plot(traces{i}(IX,1)/delaysN,traces{i}(IX,2),'.-');
            %set(h(i),'userdata',vmistats.traces{i}(:,3));%vmistats.db(vmistats.traces{i}(:,3)).fname);%vmistats.traces{i}(:,3));
            for j=1:size(traces{i},1), c{j}.fname=vmistats.db(vmistats.traces{i}(j,3)).fname; 
                                       c{j}.scan=vmistats.db(vmistats.traces{i}(j,3)).scan; end;
     %       for j=1:size(traces{i},1), c{j}=vmistats.db(vmistats.traces{i}(IX(j),3)).fname; end;
            set(h(i),'userdata',c); %store filenames in the plot itself, for retrival with Cursor
            
           end;
end; hold off;
markers='..sox+*sdv^<>ph';
colors ='bgrkcm';
for i=1:length(h), if h(i)>0,
                    set(h(i), 'marker', markers(i));
                    set(h(i), 'color', colors(1+rem(i-1,length(colors))));
                   end;
end;
legends={'111','110','101','100','011','010'};
legend(legends{inonempty});
title(pname,'interpreter','none');
%xlabel('scan #');
xlabel('image #');
ylabel('total e yield');
set(gcf,'name',[vmistats.subfolder ':   traces']);


dcm=datacursormode(gcf);
set(dcm,'updatefcn',@myfunction);

function output_txt = myfunction(obj,event_obj)
% Display the position of the data cursor
% obj          Currently not used (empty)
% event_obj    Handle to event object
% output_txt   Data cursor text string (string or cell array of strings).

pos = get(event_obj,'Position');

%------------------
xvals = get(get(event_obj,'Target'),'XData');
yvals = get(get(event_obj,'Target'),'YData');
%itrace= 8-bin2dec(get(get(event_obj,'Target'),'DisplayName'));
% now figure out which data point was selected
datapoint = find( (xvals==pos(1))&(yvals==pos(2)) );
id=get(get(event_obj,'Target'),'userdata');
fname=id{datapoint}.fname;
scan=id{datapoint}.scan;
%-------------------


% output_txt = {['X: ',num2str(pos(1),4)],...
%     ['Y: ',num2str(pos(2),4)],...
%     ['    datapoint=', num2str(datapoint)],...
%     [fname]};

%output_txt = {[fname]};

output_txt = {['X: ',num2str(pos(1),4)],...
    ['Y: ',num2str(pos(2),4)],...
    fname,...
    ['Scan: ',num2str(scan)]};

% 
% % If there is a Z-coordinate in the position, display it as well
% if length(pos) > 2
%     output_txt{end+1} = ['Z: ',num2str(pos(3),4)];
% end