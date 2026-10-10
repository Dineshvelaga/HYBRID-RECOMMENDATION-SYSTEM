import {
  benefitIcon1,
  benefitIcon2,
  benefitIcon3,
  benefitIcon4,
  benefitImage2,
  chromecast,
  disc02,
  discord,
  discordBlack,
  facebook,
  figma,
  file02,
  framer,
  homeSmile,
  instagram,
  notification2,
  notification3,
  notification4,
  notion,
  photoshop,
  plusSquare,
  protopie,
  raindrop,
  recording01,
  recording03,
  roadmap1,
  roadmap2,
  roadmap3,
  roadmap4,
  searchMd,
  slack,
  sliders04,
  telegram,
  twitter,
  yourlogo,
} from "../assets";

export const navigation = [
  {
    id: "0",
    title: "Features",
    url: "#features",
  },
  {
    id: "1",
    title: "get recommendations",
    url: "#pricing",
  },
  {
    id: "2",
    title: "speciality",
    url: "#how-to-use",
  },
  {
    id: "3",
    title: "Roadmap",
    url: "#roadmap",
  },
  
];

export const heroIcons = [homeSmile, file02, searchMd, plusSquare];

export const notificationImages = [notification4, notification3, notification2];

export const companyLogos = [yourlogo, yourlogo, yourlogo, yourlogo, yourlogo];

export const brainwaveServices = [
  "Movie recommendations",
  "Music recommendations",
  "Seamless Integration",
];

export const brainwaveServicesIcons = [
  recording03,
  recording01,
  disc02,
  chromecast,
  sliders04,
];

export const roadmap = [
  {
    id: "0",
    title: "Content based Recommendation",
    text: "Implement content-based recommendation algorithms to suggest movies and songs based on user preferences and viewing history.",
    date: "DEC 2024",
    status: "done",
    imageUrl: roadmap1,
    colorful: true,
  },
  {
    id: "1",
    title: "Collaborative filtering",
    text: "Implement collaborative filtering algorithms to suggest movies and songs based on user preferences and viewing history.",

    date: "JAN 2025",
    status: "done",
    imageUrl: roadmap2,
  },
  {
    id: "2",
    title: "Multi user recommendation",
    text: "Implement a multi-user recommendation system that can provide personalized recommendations for multiple users based on their preferences and viewing history.",
    date: "MAR 2025",
    status: "done",
    imageUrl: roadmap3,
  },
  {
    id: "3",
    title: "Integration with APIs",
    text: "Integrated with flask api to provide more relevant recommendations.",
    date: "APR 2025",
    status: "progress",
    imageUrl: roadmap4,
  },
];

export const collabText =
  "With smart Integration of movie and music recommendation, you can easily find the best content for your needs. Our app uses advanced algorithms to analyze your preferences and provide personalized recommendations that are tailored to you.";

export const collabContent = [
  {
    id: "0",
    title: "Seamless Integration",
    
  },
  {
    id: "1",
    title: "Best Recommendations",
  },
  {
    id: "2",
    title: "Multi user recommendations",
  },
];

export const collabApps = [
  {
    id: "0",
    title: "Figma",
    icon: figma,
    width: 26,
    height: 36,
  },
  {
    id: "1",
    title: "Notion",
    icon: notion,
    width: 34,
    height: 36,
  },
  {
    id: "2",
    title: "Discord",
    icon: discord,
    width: 36,
    height: 28,
  },
  {
    id: "3",
    title: "Slack",
    icon: slack,
    width: 34,
    height: 35,
  },
  {
    id: "4",
    title: "Photoshop",
    icon: photoshop,
    width: 34,
    height: 34,
  },
  {
    id: "5",
    title: "Protopie",
    icon: protopie,
    width: 34,
    height: 34,
  },
  {
    id: "6",
    title: "Framer",
    icon: framer,
    width: 26,
    height: 34,
  },
  {
    id: "7",
    title: "Raindrop",
    icon: raindrop,
    width: 38,
    height: 32,
  },
];

export const pricing = [
  {
    id: "0",
    title: "Movie Model",
    description: "ML based Movie Recommendations",
    price: "3 Models",
    features: [
      "A Hybrid Movie model for better recommendations",
      "Personalized recommendations based on your preferences",
      "Multi user recommendations",
    ],
  },
  {
    id: "1",
    title: "Music Model",
    description: "ML based Song Recommendations",
    price: "4 Models",
    features: [
      "An advanced ML model that can recommend songs",
      "Hybrid song recommendations based on your preferences",
      "4 different models for better recommendations",
    ],
  },
  
];

export const benefits = [
  {
    id: "0",
    title: "Personalized recommendations",
    text: "The app uses ML Algorithms to understand user queries and provide accurate and relevant responses.",
    backgroundUrl: "./src/assets/benefits/card-1.svg",
    iconUrl: benefitIcon1,
    imageUrl: benefitImage2,
  },
  {
    id: "1",
    title: "Recommend variety of contents",
    text: "The app can be able to recommend a variety of contents, including movies and music.",
    backgroundUrl: "./src/assets/benefits/card-2.svg",
    iconUrl: benefitIcon2,
    imageUrl: benefitImage2,
    light: true,
  },
  {
    id: "2",
    title: "Find Popular contents",
    text: "Popular movies and music can be recommended to users.",
    backgroundUrl: "./src/assets/benefits/card-3.svg",
    iconUrl: benefitIcon3,
    imageUrl: benefitImage2,
  },
  {
    id: "3",
    title: "Fast responding",
    text: "Lets users quickly find their recommendations.",
    backgroundUrl: "./src/assets/benefits/card-4.svg",
    iconUrl: benefitIcon4,
    imageUrl: benefitImage2,
    light: true,
  },
  {
    id: "4",
    title: "Multi user recommendation",
    text: "The app can recommend content to multiple users effectively.",
    backgroundUrl: "./src/assets/benefits/card-5.svg",
    iconUrl: benefitIcon1,
    imageUrl: benefitImage2,
  },
  {
    id: "5",
    title: "new user friendly",
    text: "The app is designed to be user-friendly and easy to navigate and get recommendations.",
    
    backgroundUrl: "./src/assets/benefits/card-6.svg",
    iconUrl: benefitIcon2,
    imageUrl: benefitImage2,
  },
];

export const socials = [
  {
    id: "0",
    title: "Discord",
    iconUrl: discordBlack,
    url: "#",
  },
  {
    id: "1",
    title: "Twitter",
    iconUrl: twitter,
    url: "#",
  },
  {
    id: "2",
    title: "Instagram",
    iconUrl: instagram,
    url: "#",
  },
  {
    id: "3",
    title: "Telegram",
    iconUrl: telegram,
    url: "#",
  },
  {
    id: "4",
    title: "Facebook",
    iconUrl: facebook,
    url: "#",
  },
];
