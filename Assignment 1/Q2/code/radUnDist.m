function imOut = radUnDist(imIn, k1, k2, nSteps)
    % Your code here
    [m, n] = size(imIn);
    [x, y] = meshgrid(1:n, 1:m);
    k = 1;
    cx = m/2;
    cy = n/2;
    
    x = x - cx;
    y = y - cy;
    x = x/cx;
    y = y/cy;

    x_dist = x;
    y_dist = y;

    for i = 1:nSteps
    	r2 = sqrt(x.^2 + y.^2);
    	dr = 1 - k1*r2 - k2*r2.^2;
    	x = x_dist.*dr;
    	y = y_dist.*dr;
	end

	x = x*cx + cx;
    y = y*cy + cy;

    imOut = interp2(imIn, x, y, 'cubic');

end