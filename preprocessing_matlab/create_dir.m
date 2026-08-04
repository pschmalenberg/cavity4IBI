function [] = create_dir(dir)


    if ~exist(dir)
        disp(strcat('creating folder: ', dir));
        mkdir(dir);
    else
        disp(strcat('folder already exists: ', dir))
    end 



end