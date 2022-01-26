// This is our bucket for storing data. Access is set to private and leveraging lifecycle rules, we can expire our objects base on their prefix.

resource "aws_s3_bucket" "bucket" {
  bucket = "covcough-${var.deploymentname}"
  acl    = "private"

  lifecycle_rule {
    id      = "catchall"
    enabled = true
    expiration {
      days = 60
    }
  }

   cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET","PUT", "POST"]
    allowed_origins = ["*"]
    expose_headers = ["x-amz-meta-tag"]
    max_age_seconds = 2000
  }

   server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm     = "AES256"
      }
    }
  }

}

resource "aws_s3_bucket_public_access_block" "bucket" {
  bucket                  = aws_s3_bucket.bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket_notification" "processupload" {
    bucket = aws_s3_bucket.bucket.id
    lambda_function {
        lambda_function_arn = aws_lambda_function.processupload.arn
        events              = ["s3:ObjectCreated:*"]
        filter_prefix       = "records/"
    }
}

locals {
  processuploadenvar = {
    "APIGATEWAY_LAMBDA" = "${aws_api_gateway_deployment.covcough.invoke_url}${aws_api_gateway_stage.covcough_alpha.stage_name}"
    "NUMBA_CACHE_DIR" = "/tmp"
    "LD_LIBRARY_PATH" = "/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib"
  }
}


// processupload lambda function, reuse covcough lambda role to get file metadata
resource "aws_lambda_function" "processupload" {
  function_name = "processupload-${var.deploymentname}-function"

  image_uri = var.processuploadimageurl
  package_type = "Image"
  memory_size = 3008
  timeout = 300
  role    = aws_iam_role.covcough.arn

  environment {
    variables = merge(var.envvar,local.extraenvar,local.processuploadenvar)
  }
}

// Allow s3 bucket to trigger/invoke processupload function
resource "aws_lambda_permission" "createnofity_s3" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.processupload.arn
  principal = "s3.amazonaws.com"
  source_arn = aws_s3_bucket.bucket.arn
}

// processupload lambda cloudwatch log
resource "aws_cloudwatch_log_group" "processupload" {
  name              = "/aws/lambda/${aws_lambda_function.processupload.function_name}"
  retention_in_days = 5
}
