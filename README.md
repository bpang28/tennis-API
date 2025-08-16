## RUNNING

This is a copy of the API code in EC2.

The EC2 is connected to a load balancer that always keeps [this site](http://tennis-lb-api-308609498.us-east-1.elb.amazonaws.com/docs#/default) alive.

To update the API, pull these files, make your changes, copy them to the EC2 instance, ssh into the instance, activate the .venv, and run this code segment:

sudo systemctl daemon-reload

sudo systemctl restart fastapi.service

sudo systemctl status fastapi.service


You can track debug statements by running this line as well:

sudo journalctl -u fastapi.service -f


## MODELS

All the tracking models have already been copied to the EC2, but if you need them, you can refer to the README at [this page](https://github.com/deep-dive-mexico/tennis-poc)
