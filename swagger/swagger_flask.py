from flask import Flask
from flask_restx import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage



app = Flask(__name__)

api = Api(app, version='1.0', title='API 문서', description='Swagger 문서', doc="/api-docs")

test_api = api.namespace('test', description='조회 API')

upload_parser = test_api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@test_api.route('/upload/')
@test_api.expect(upload_parser)
class Upload(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        url = do_something_with_file(uploaded_file)
        return {'url': url}, 201


def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'


'''
@test_api.route('/')
class Test(Resource):
    def get(self):
    	return 'Hello World!'
'''


if __name__ == '__main__':
    app.run()