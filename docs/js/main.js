/// PWA app component
///
//Register the service worker.
// if('serviceWorker' in navigator) {
// 	navigator.serviceWorker
// 	.register('../sw.js')
// 	.then(function() {
// 		console.log("Service Worker registered successfully");
// 	})
// 	.catch(function() {
// 		console.log("Service worker registration failed")
// 	});
// }

// // Check if this is an IOS device and if the webpage is not yet installed as progressive app
// function isIos() {
// 	const userAgent = window.navigator.userAgent.toLowerCase();
// 	return /iphone|ipad|ipod/.test( userAgent );
//   }
//   // Detects if device is in standalone mode
//   const isInStandaloneMode = () => ('standalone' in window.navigator) && (window.navigator.standalone);
//   // Checks if should display install popup notification:
//   if (isIos() && !isInStandaloneMode()) {
// 	document.getElementById('addToHomescreenGadget').style.display='block'
// 	this.setState({ showInstallMessage: true });
// 	setTimeout(function () {
// 	  document.getElementById('addToHomescreenGadget').style.display='none'
// 	}, 5000)
// } 
  
///////////// MAIN APP starts here //////////////////

var audioStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording
demosite = false;


function bufferToHex (buffer) {
    return [...new Uint8Array (buffer)]
        .map (b => b.toString (16).padStart (2, "0"))
        .join ("");
}

async function digestMessage(message) {
	const encoder = new TextEncoder();
	const data = encoder.encode(message);
	const hash = await crypto.subtle.digest('SHA-256', data);
	return bufferToHex(hash);
}

demo=false

async function asyncmain() {
	demositehex=["bf4d27b7f53fcc290452b26a924a735029763a09ecab30d24ee4f5356e93597c","d810340b4366d74b219f2faf81bd1772610d7cd78a393fe8dda0b015a97fafaa"]
	currentsitehash = await digestMessage(document.location.host)
	if (document.location.origin.indexOf("localhost") != -1  || demositehex.includes(currentsitehash)){
		demo=true
	}
	// Check if we are on demosite or localhost instance. If we are, disable name input and use "demositedata"
	if (demo) {
		document.getElementById("name").value = "demositedata_pleaseignore"
		document.getElementById("name").style.display = "none";
		demosite = true;
	}
}

asyncmain()

// Check whether we have access to Microphone or not. If not, block access to the app and show modal overlay.
navigator.mediaDevices.getUserMedia({ audio: true })
	.then(function (stream) {
		console.log('We have microphone permission!')
	})
	.catch(function (err) {
		console.log('No microphone permission!');
		document.getElementById("helptext").style.display="none";
		document.getElementById("modal").removeAttribute("onclick");
		openmodal()
		updatemodaltext(nomictxt, false)
	});

/////////////// AUDIO recording component ///////////////////
// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var brec = document.getElementById("brec");
var bstop = document.getElementById("bstop");
var bpause = document.getElementById("bpause");

//add events to those 2 buttons
brec.addEventListener("click", startRecording);
bstop.addEventListener("click", stopRecording);
bpause.addEventListener("click", pauseRecording);


function startRecording() {
	console.log("brec clicked");
	var constraints = { audio: true, video: false }
	brec.disabled = true;
	bstop.disabled = false;
	bpause.disabled = false

	navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
		console.log("Initializing Recorder.js ...");
		audioContext = new AudioContext();
		audioStream = stream;
		input = audioContext.createMediaStreamSource(stream);
		rec = new Recorder(input, { numChannels: 1 })

		//start the recording process
		rec.record()
		console.log("Recording started");
		setTimeout(function () {
			stopRecording()
		}, 15000)

	}).catch(function (err) {
		brec.disabled = false;
		bstop.disabled = true;
		bpause.disabled = true
	});
}

function pauseRecording() {
	console.log("bpause clicked");
	if (rec.recording) {
		//pause
		rec.stop();
		bpause.innerHTML = "Resume";
	} else {
		//resume
		rec.record()
		bpause.innerHTML = "Pause";

	}
}

function stopRecording() {
	if (!rec.recording) {
		console.log("recording is already stopped")
		return
	}
	console.log("bstop clicked");
	document.getElementById("statuschoices").classList.add("hidden");

	//disable the stop button, enable the record too allow for new recordings
	bstop.disabled = true;
	brec.disabled = false;
	bpause.disabled = true;

	//reset button just in case the recording is stopped while paused
	bpause.innerHTML = "Pause";

	//tell the recorder to stop the recording
	rec.stop();

	//stop microphone access
	audioStream.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);
}

///////////////// INTERACT WITH COVCOUGH API //////////////////

async function pollresult(resultjson, counter) {
	counter += 1;
	pngresult = resultjson.pngresult;
	jsonresult = resultjson.jsonresult;
	let response = await fetch(jsonresult.signedurl)
	if (counter > 60) {
		updatemodaltext("Something went wrong... Please try again", false)
		return
	}
	if (response.status != 200) {
		await new Promise(resolve => setTimeout(resolve, 1000));
		await pollresult(resultjson, counter);
	} else {
		data = await response.json();
		if (data.Result.length == 0){  // If JSON result does not return an array - we have an error.
			updatemodaltext(nocoughtxt);
		} else {
			// updatemodaltext(resulttxt + data.Result)
			updatemodaltext(resulttxt);
			document.getElementById('acceptshowresult').style.display='block'
			updatemodalimage(pngresult.signedurl);
		}
	}
}

async function uploadToS3_default(datablob, originalfilename, status) {
	openmodal()
	document.getElementById("modal").removeAttribute("onclick");
	document.getElementById("helptext").style.display="none"
	updatemodaltext("Getting presigned key to upload ...", true)
	console.log("Getting presigned s3 URL for upload.");
	// If a timetoken exist, include it.
	var url = lambdaurl + '/upload/' + status;
	token = getUrlVars()["token"];
	if (token != undefined) {
		url += "?token=" + token
	}
	
	var filemetadata = {
		name: originalfilename
	}

	try {
		response = await fetch(url);
		if (response.ok) {
			responsejson = await response.json()
		}
		else {
			updatemodaltext("Chương trình lấy mẫu vẫn đang trong thời gian thử nghiệm, Xin đừng chia xẻ và chỉ dùng link trên một thiết bị!", false)
			console.log("Error getting presigned key for upload")
			// closemodal();
			return false
		}

		signedupload = responsejson.signedupload

		const formData = new FormData();
		formData.append("Content-Type", "audio/wav");
		formData.append("x-amz-meta-tag", (JSON.stringify(filemetadata)))
		Object.entries(signedupload.fields).forEach(([k, v]) => {
			formData.append(k, v);
		});
		formData.append("file", datablob);
		updatemodaltext("Uploading to S3 ...", true)
		response = await fetch(
			signedupload.url,
			{
				method: "POST",
				body: formData,
			})

		if (response.status == 204) {
			updatemodaltext("Please wait for result ...", true);
			console.log("File successfully uploaded!");
			pollresult(responsejson, 0);
			// document.body.classList.remove("showmodal");
			return true
		} else {
			console.log("Fail to upload file!")
			closemodal();
			return false
		}
	}
	catch (err) {
		console.log("Failed to upload"); // This is where you run code if the server returns any errors
		console.log(err);
		return false
	}
}


async function uploadToS3_individual(datablob, originalfilename, status) {
	openmodal()
	document.getElementById("modal").removeAttribute("onclick");
	document.getElementById("helptext").style.display="none"
	updatemodaltext("Getting presigned key to upload ...", true)
	console.log("Getting presigned s3 URL for upload.");
	// Include the individual token
	var url = lambdaurl + '/upload/' + status +"?token=" + token;
	var filemetadata = {
		name: originalfilename
	}

	try {
		response = await fetch(url);
		if (response.ok) {
			responsejson = await response.json()
		}
		else {
			updatemodaltext("Chương trình lấy mẫu vẫn đang trong thời gian thử nghiệm, Xin đừng chia xẻ và chỉ dùng link trên một thiết bị!", false)
			console.log("Error getting presigned key for upload")
			// closemodal();
			return false
		}

		signedupload = responsejson.signedupload
		verifieddevice = true;
		if (signedupload.fields.key.match("sample1_")){
			console.log("New device - finger print")
			fingerprint(token);
		} else {
			console.log("Old device - verify finger print")
			verifieddevice = checkfingerprint(token);
		}
		if (!verifieddevice){
			updatemodaltext("Chương trình lấy mẫu vẫn đang trong thời gian thử nghiệm, Xin đừng chia xẻ và chỉ dùng link trên một thiết bị!", false)
			return
		}
		const formData = new FormData();
		formData.append("Content-Type", "audio/wav");
		formData.append("x-amz-meta-tag", (JSON.stringify(filemetadata)))
		Object.entries(signedupload.fields).forEach(([k, v]) => {
			formData.append(k, v);
		});
		formData.append("file", datablob);
		updatemodaltext("Uploading to S3 ...", true)
		response = await fetch(
			signedupload.url,
			{
				method: "POST",
				body: formData,
			})

		if (response.status == 204) {
			updatemodaltext("Please wait for result ...", true);
			console.log("File successfully uploaded!");
			pollresult(responsejson, 0);
			// document.body.classList.remove("showmodal");
			return true
		} else {
			console.log("Fail to upload file!")
			closemodal();
			return false
		}
	}
	catch (err) {
		console.log("Failed to upload"); // This is where you run code if the server returns any errors
		console.log(err);
		return false
	}
}

uploadToS3 = uploadToS3_default

if (document.cookie.match("id:[a-zA-Z0-9\/\.]*")!=null){
	console.log("Found individual token in the cookie")
	token = document.cookie.match("id:[a-zA-Z0-9\/\.]*")[0]
	uploadToS3 = uploadToS3_individual
}


if (getUrlVars()["token"] != undefined) {
	token = getUrlVars()["token"];
	// If this is an individual token
	if (token.startsWith("id:")) {
		uploadToS3 = uploadToS3_individual
	}
}




function createDownloadLink(blob) {

	var url = window.URL.createObjectURL(blob);
	var au = document.createElement('audio');
	var li = document.createElement('li');
	var link = document.createElement('a');

	//name of .wav file to use during upload and download (without extendion)
	var t = new Date().getTime().toString()
	if (document.getElementById("name").value != "") {
		filename = document.getElementById("name").value + "_";
	} else {
		filename = "anon_";
	}
	status = document.querySelector("input[name=status]:checked").value;
	if (demosite == true) {
		status = "demosite"
	}
	filename += status + "_"
	filename += t
	au.controls = true;
	au.src = url;

	//save to disk link
	//link.href = url;
	//link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
	//link.innerHTML = "Save to disk";

	//add the new audio element to li
	li.appendChild(au);
	li.appendChild(document.createElement('br'));

	//add the filename to the li
	li.appendChild(document.createTextNode(filename + ".wav "))
	li.appendChild(document.createElement('br'));

	//add the save to disk link to li
	li.appendChild(link);

	//upload link
	var upload = document.createElement('button');
	upload.href = "#";
	upload.innerHTML = sendfilebuttontxt;
	upload.addEventListener("click", async function (event) {
		uploadurl = event.target;
		parentnode = uploadurl.parentElement;
		parentnode.removeChild(uploadurl)
		sp = document.createElement("span")
		parentnode.appendChild(sp)
		sp.innerHTML = "Please wait..."
		uploaded = await uploadToS3(blob, filename, status)
		if (uploaded) {
			sp.innerHTML = filesenttxt;
			parentnode.appendChild(sp)
		} else {
			sp.innerHTML = "Error... Please try again later."
			parentnode.appendChild(sp)
		}
	})
	li.appendChild(document.createTextNode(" "))//add a space in between
	li.appendChild(upload)//add the upload link to li

	//add the li element to the ol
	records.appendChild(li);
}


//////////////// HELPER FUNCTIONS /////////////////


function getUrlVars() {
    urlwithoutanchor=window.location.href.split("#")[0]
    var vars = {};
    var parts = urlwithoutanchor.replace(/[?&]+([^=&]+)=([^&]*)/gi, function (m, key, value) {
        vars[key] = value;
    });
    return vars;
}

// A super simple fingerprint just to make it harder to share the URL
function fingerprint(token) {
	document.cookie = "token="+token+"; expires=Tue, 01 Jan 2099 00:00:00 UTC; path=/"
}

// Just simply check if the token exist in the cookie. Can be bypass easily but is acceptable risk
function checkfingerprint(token) {
	if (document.cookie.match(token)){
		return true
	} else {
		return false
	}
}

function updatemodalimage(url) {
	modalimage = document.getElementById("modalimage");
	modalimage.innerHTML = "";
	img = document.createElement('img');
	img.src = url
	img.style.cssText = "padding-left: 5%;max-width: 90%;"
	modalimage.appendChild(img);
}

function updatemodaltext(text, flashing = false) {
	modaltext = document.getElementById("modalstatustext")
	if (flashing) {
		modaltext.classList.add("loadingtext");
	} else {
		modaltext.classList.remove("loadingtext");
	}
	modaltext.innerText = text;
}

function closemodal(){
	document.body.classList.remove('showmodal')
}

function openmodal(){
	document.body.classList.add('showmodal')
}
