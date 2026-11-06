AWS Batch configuration
=======================

Replace ``ACCOUNT_ID`` with your AWS account ID before using these files. Don't commit the result.

* ``job-definition.json``: Batch job definition for ``martini-run`` on Fargate.
* ``job-role-trust.json``: lets ECS tasks in the account assume the job role.
* ``job-role-policy.json``: S3 access for the job role, limited to the ``martini/`` prefix.
