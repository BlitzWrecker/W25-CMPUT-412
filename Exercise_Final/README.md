dts devel run -H csc22928 -L parking -- --env PARKING_STALL=3

# To shutdown the bot after parking
dts devel run -H csc22928 -L parking -- --env PARKING_STALL=3 && dts duckiebot shutdown csc22928
