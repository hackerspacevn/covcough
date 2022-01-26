// Change to your lambda endpoint here
var lambdaurl = 'https://api.covcough.com';

// Temporary redirect when the app is still in development.
if (document.location.origin.indexOf("localhost") == -1 && document.location.origin.indexOf("192.168") == -1  && document.location.origin.indexOf("surge.sh") == -1){
	// document.location = "./underconstruction.html";
}
