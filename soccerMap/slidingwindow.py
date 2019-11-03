def cut_frame(frame, window_height=128, window_width=48, increment=5):
	x_window = 0
	y_window = 0
	while x_window + window_width <= frame.shape[1]:
		x_window += increment
		while y_window + window_height <= frame.shape[0]:
			y_window += increment
			yield (frame[y_window : y_window + window_height, x_window : x_window + window_width], y_window, x_window)
			